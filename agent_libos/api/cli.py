from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import threading
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig, load_config_file, load_config_from_project_root
from agent_libos.evidence import (
    PayloadRetentionCursor,
    PayloadRetentionKind,
    PayloadRetentionRequest,
)
from agent_libos.models import (
    CapabilityEffect,
    CapabilityRight,
    CapabilitySpec,
    ForkMode,
    HumanRequestStatus,
    MemoryViewSpec,
    ObjectHandle,
    ObjectMetadata,
    ObjectRight,
    ObjectType,
    ProcessMessage,
    ProcessMessageKind,
    ProcessStatus,
    ToolCallResult,
    ViewMode,
    process_state_to_mapping,
)
from agent_libos.models.exceptions import HumanApprovalRequired, LibOSError, NotFound
from agent_libos.models.exceptions import ValidationError as LibOSValidationError
from agent_libos.modules import ModuleLoader
from agent_libos.runtime.runtime import Runtime
from agent_libos.storage import display_store_target
from agent_libos.utils.serde import bounded_json_loads, to_jsonable

_RUNTIME_DEFAULTS = DEFAULT_CONFIG.runtime
_WORKFLOW_HELP = "Run an Image-bound workflow tool directly without an LLM turn"
_WORKFLOW_RUN_HELP = "Spawn a workflow process and call one tool from its complete process table"
_TASK_RUN_STATUSES = frozenset(
    {
        "queued",
        "running",
        "waiting_human",
        "waiting_process",
        "waiting_message",
        "waiting_tool",
        "paused",
        "cancelling",
        "finalizing",
        "needs_attention",
        "succeeded",
        "failed",
        "cancelled",
    }
)
_TASK_RUN_RETENTIONS = ("purge_on_terminal", "permanent")

DEMO_PATCH_PREVIEW_PATH = "agent_outputs/demo_patch_preview.txt"
DEMO_PATCH_PREVIEW_CONTENT = "change add() expected value\n"
_TERMINAL_PROCESS_STATUSES = {ProcessStatus.EXITED, ProcessStatus.FAILED, ProcessStatus.KILLED}
_COMPACT_COMMANDS = frozenset(
    {
        "audit",
        "llm-calls",
        "payload-retention",
        "processes",
        "resources",
        "tools",
    }
)
_PRIMARY_COMMANDS = frozenset(
    {"init", "demo", "explain", "workflow", "object-task", "task-run"}
)
_PROCESS_COMMANDS = frozenset(
    {"spawn", "cd", "exec", "exit", "llm-once", "run", "message", "interrupt"}
)
_MANAGEMENT_COMMANDS = frozenset(
    {
        "checkpoint",
        "skills",
        "capabilities",
        "images",
        "jsonrpc",
        "mcp",
        "modules",
        "human",
    }
)


def _load_runtime_config(config_path: str | None, parser: argparse.ArgumentParser) -> AgentLibOSConfig:
    try:
        if config_path:
            return load_config_file(config_path)
        return load_config_from_project_root()
    except (OSError, ValueError, PydanticValidationError) as exc:
        parser.error(str(exc))
    raise AssertionError("argparse parser.error should exit")


def _run_payload_retention_command(
    runtime: Runtime,
    args: argparse.Namespace,
) -> Any:
    cursor_values = (args.after_created_at, args.after_record_id)
    if (cursor_values[0] is None) != (cursor_values[1] is None):
        raise LibOSValidationError(
            "payload retention cursor requires both --after-created-at and "
            "--after-record-id"
        )
    try:
        cursor = (
            PayloadRetentionCursor(*cursor_values)
            if cursor_values[0] is not None and cursor_values[1] is not None
            else None
        )
        return runtime.payload_retention.run(
            PayloadRetentionRequest(
                kind=PayloadRetentionKind(args.kind),
                dry_run=not args.apply,
                cursor=cursor,
                limit=args.limit,
                actor=args.actor,
                correlation_id=args.correlation_id,
            )
        )
    except ValueError as exc:
        raise LibOSValidationError(str(exc)) from exc


def _process_to_mapping(process: Any) -> dict[str, Any]:
    """Serialize an AgentProcess without dropping its tagged state fields."""

    return {
        **to_jsonable(process),
        **process_state_to_mapping(
            process.status.value,
            process.wait_state,
            process.outcome,
            process.state_generation,
        ),
    }


def _add_payload_retention_parser_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "kind",
        choices=[item.value for item in PayloadRetentionKind],
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the page; the default is a non-mutating dry run.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--after-created-at")
    parser.add_argument("--after-record-id")
    parser.add_argument("--actor", default="host.retention")
    parser.add_argument("--correlation-id")


def _run_compact_command(
    runtime: Runtime,
    args: argparse.Namespace,
) -> Any:
    if args.command == "audit":
        return [record.__dict__ for record in runtime.audit.trace()]
    if args.command == "llm-calls":
        return [
            record.__dict__
            for record in runtime.store.list_llm_calls(
                pid=args.pid,
                limit=args.limit,
            )
        ]
    if args.command == "payload-retention":
        return _run_payload_retention_command(runtime, args)
    if args.command == "processes":
        return [_process_to_mapping(process) for process in runtime.process.list()]
    if args.command == "resources":
        return _resource_summary(runtime, args.pid)
    if args.command == "tools":
        return runtime.tools.list()
    raise AssertionError(f"unsupported compact command: {args.command}")


def _shutdown_runtime_before_exit(runtime: Runtime) -> None:
    """Require the host Runtime to report a complete CLI teardown."""

    result = runtime.shutdown(actor="cli", reason="cli.command_complete")
    try:
        completed = result["ok"] is True
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "CLI runtime teardown returned an invalid result without an 'ok' status"
        ) from exc
    if not completed:
        raise RuntimeError(
            "CLI runtime teardown remained incomplete: "
            f"{to_jsonable(result)!r}"
        )


def _parse_cli_args(
    argv: list[str] | None,
) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(prog="agent-libos")
    parser.add_argument(
        "--config",
        help="YAML config overlay. Defaults to the project-root config.yaml when present.",
    )
    parser.add_argument(
        "--db",
        help=(
            "Runtime store target. Paths/local use SQLite; postgresql:// DSNs use PostgreSQL. "
            "If omitted, uses the selected config runtime store settings "
            f"(default '{_RUNTIME_DEFAULTS.local_store_target}' is in-memory SQLite)."
        ),
    )
    parser.add_argument(
        "--module-manifest",
        action="append",
        default=[],
        help="Trusted startup module manifest to load before the runtime is used. May be passed multiple times.",
    )
    parser.add_argument(
        "--trusted-module",
        action="append",
        default=[],
        help=(
            "Trusted startup module entry in the form "
            "'<module_id>:<manifest_sha256>:<source_sha256>'. Use the trust_key from `modules verify`."
        ),
    )
    parser.add_argument(
        "--trusted-module-sha256",
        action="append",
        default=[],
        help=(
            "Trusted startup module digest pair in the form '<manifest_sha256>:<source_sha256>', "
            "regardless of module id. Intended for local development only."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Initialize a runtime database")
    sub.add_parser("demo", help="Run the deterministic coding-agent demo")
    sub.add_parser("audit", help="Print audit trace")
    explain_parser = sub.add_parser("explain", help="Explain protected operations from explicit runtime evidence")
    _add_explain_parser_args(explain_parser)
    llm_calls_parser = sub.add_parser("llm-calls", help="Print persisted LLM call records")
    llm_calls_parser.add_argument("--pid", help="Filter by process id.")
    llm_calls_parser.add_argument("--limit", type=int, help="Maximum number of records to print.")
    retention_parser = sub.add_parser(
        "payload-retention",
        help="Preview or apply one bounded LLM/effect payload-retention page",
    )
    _add_payload_retention_parser_args(retention_parser)
    sub.add_parser("processes", help="Print process table")
    resources_parser = sub.add_parser("resources", help="Print process resource budget and usage")
    resources_parser.add_argument("pid")
    sub.add_parser("tools", help="Print registered tools")
    workflow_parser = sub.add_parser("workflow", help=_WORKFLOW_HELP)
    _add_workflow_parser_args(workflow_parser)
    object_task_parser = sub.add_parser(
        "object-task",
        help="Start, get, list, cancel, wait for, or watch Object tasks",
    )
    _add_object_task_parser_args(object_task_parser)
    task_run_parser = sub.add_parser(
        "task-run",
        help="Create, supervise, recover, or rerun a durable TaskRun",
    )
    _add_task_run_parser_args(task_run_parser)
    spawn_parser = sub.add_parser("spawn", help="Spawn a process")
    spawn_parser.add_argument("--image")
    spawn_parser.add_argument("--goal", required=True)
    spawn_parser.add_argument("--llm-profile", help="Optional host-selected LLM profile id for the new process.")
    _add_spawn_authority_manifest_parser_arg(spawn_parser)
    cd_parser = sub.add_parser("cd", help="Set an AgentProcess working directory")
    cd_parser.add_argument("pid")
    cd_parser.add_argument("path")
    exec_parser = sub.add_parser("exec", help="Exec an AgentProcess into another image")
    exec_parser.add_argument("image", help="Target AgentImage id, or an image package directory containing IMAGE.yaml.")
    exec_parser.add_argument("goal", help="Replacement process goal.")
    exec_parser.add_argument("--pid", required=True, help="Process id to exec.")
    exec_parser.add_argument("--llm-profile", help="Optional host-selected LLM profile id for the existing process.")
    exec_parser.add_argument("--replace-image", action="store_true", help="Allow an image package to replace an existing image id.")
    exec_parser.add_argument("--args-json", default="{}", help="JSON object recorded as structured exec args.")
    exec_parser.add_argument(
        "--preserve-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the current MemoryView across exec. Use --no-preserve-memory to replace it with the new goal only.",
    )
    exec_parser.add_argument(
        "--preserve-capabilities",
        action="store_true",
        help="Keep existing external capabilities. Exec never grants target image required_capabilities automatically.",
    )
    run_group = exec_parser.add_mutually_exclusive_group()
    run_group.add_argument("--run", dest="run", action="store_true", default=False, help="Run the scheduler after exec.")
    run_group.add_argument("--no-run", dest="run", action="store_false", help="Only apply exec; do not run the scheduler.")
    exec_parser.add_argument("--max-quanta", type=int, help="Optional quantum budget when --run is set; omitted uses runtime.run_until_idle_max_quanta (which may be unbounded).")
    exit_parser = sub.add_parser("exit", help="Exit an AgentProcess")
    exit_parser.add_argument("pid")
    exit_parser.add_argument(
        "--message",
        help="Optional final message stored in a label-bearing process-result Object.",
    )
    exit_parser.add_argument("--payload", help="Optional JSON final-result payload. Non-JSON text is wrapped as content.")
    exit_parser.add_argument("--result-oid", help="Existing object id to use as process result.")
    exit_parser.add_argument("--failed", action="store_true", help="Mark the process as failed instead of exited.")
    llm_once_parser = sub.add_parser("llm-once", help="Run one LLM quantum for a process")
    llm_once_parser.add_argument("pid")
    run_parser = sub.add_parser("run", help="Run runnable processes with the LLM scheduler")
    run_parser.add_argument("--max-quanta", type=int, help="Optional quantum budget; omitted uses runtime.run_until_idle_max_quanta (which may be unbounded).")
    run_parser.add_argument("--interactive", action="store_true", help="Read human input while running and post it as process messages.")
    run_parser.add_argument("--pid", help="Default target process for interactive human messages.")
    run_parser.add_argument("--human", help="Human actor name for interactive messages.")
    run_parser.add_argument("--message-channel", default="human", help="Process-message channel for interactive human input.")
    message_parser = sub.add_parser("message", help="Send a human process message")
    _add_message_parser_args(message_parser)
    interrupt_parser = sub.add_parser("interrupt", help="Send a human interrupt process message")
    _add_message_parser_args(interrupt_parser, include_kind=False)
    checkpoint_parser = sub.add_parser(
        "checkpoint",
        help="Create, list, inspect, diff, restore, fork, or replay checkpoints",
    )
    _add_checkpoint_parser_args(checkpoint_parser)
    skills_parser = sub.add_parser(
        "skills",
        help=(
            "Discover, inspect, validate, register, trust, untrust, activate, "
            "or unload skills"
        ),
    )
    _add_skills_parser_args(skills_parser)
    capabilities_parser = sub.add_parser("capabilities", help="List, inspect, grant, delegate, revoke, or explain capabilities")
    _add_capabilities_parser_args(capabilities_parser)
    images_parser = sub.add_parser("images", help="List, inspect, validate, register, or commit AgentImages")
    _add_images_parser_args(images_parser)
    jsonrpc_parser = sub.add_parser(
        "jsonrpc",
        help="Register, list, inspect, call, or unregister JSON-RPC over HTTP endpoints",
    )
    _add_jsonrpc_parser_args(jsonrpc_parser)
    mcp_parser = sub.add_parser(
        "mcp",
        help="Register, list, inspect, discover, enumerate tools, call, or unregister MCP servers",
    )
    _add_mcp_parser_args(mcp_parser)
    modules_parser = sub.add_parser("modules", help="List, inspect, or verify startup runtime modules")
    _add_modules_parser_args(modules_parser)
    sub.add_parser("human", help="Process pending human messages in terminal order")
    return parser, parser.parse_args(argv)


def _run_primary_cli_command(
    runtime: Runtime,
    args: argparse.Namespace,
    selected_db_display: str,
) -> None:
    if args.command == "init":
        print(f"initialized {selected_db_display}")
        return
    if args.command == "demo":
        print(json.dumps(run_demo(runtime), indent=2, ensure_ascii=False))
        return
    if args.command == "explain":
        try:
            result = _run_explain_command(runtime, args)
        except NotFound as exc:
            _print_json(
                {
                    "schema_version": 1,
                    "error": {"type": "NotFound", "message": str(exc)},
                }
            )
            raise SystemExit(1) from None
        _print_json(result)
        if result.get("ambiguous"):
            raise SystemExit(2)
        return
    if args.command == "workflow":
        result = _run_workflow_command(runtime, args)
        _print_json(to_jsonable(result))
        if not result.ok:
            raise SystemExit(1)
        return
    if args.command == "object-task":
        _print_json(_run_object_task_command(runtime, args))
        return
    if args.command == "task-run":
        _print_json(_run_task_run_command(runtime, args))
        return
    raise AssertionError(f"unsupported primary command: {args.command}")


def _run_process_cli_command(runtime: Runtime, args: argparse.Namespace) -> None:
    if args.command == "spawn":
        pid = runtime.process.spawn(
            image=args.image,
            goal=args.goal,
            llm_profile_id=args.llm_profile,
            authority_manifest=(
                _parse_json_mapping(
                    args.authority_manifest_json,
                    "--authority-manifest-json",
                )
                if args.authority_manifest_json is not None
                else None
            ),
        )
        process = runtime.process.get(pid)
        _print_json(
            {
                "pid": pid,
                "image": process.image_id,
                "llm_profile_id": process.llm_profile_id,
                "goal": args.goal,
            }
        )
        return
    if args.command == "cd":
        _print_json(_run_cd_command(runtime, args))
        return
    if args.command == "exec":
        _print_json(asyncio.run(_run_exec_command(runtime, args)))
        return
    if args.command == "exit":
        _print_json(_run_exit_command(runtime, args))
        return
    if args.command == "llm-once":
        _print_json(asyncio.run(runtime.arun_process_once(args.pid)))
        return
    if args.command == "run":
        if args.interactive:
            result = asyncio.run(_run_interactive_command(runtime, args))
        else:
            result = asyncio.run(
                runtime.arun_until_idle(max_quanta=args.max_quanta)
            )
        _print_json(result)
        return
    if args.command == "message":
        _print_json(asyncio.run(_run_message_command(runtime, args)))
        return
    if args.command == "interrupt":
        _print_json(
            asyncio.run(
                _run_message_command(
                    runtime,
                    args,
                    fixed_kind=ProcessMessageKind.INTERRUPT,
                )
            )
        )
        return
    raise AssertionError(f"unsupported process command: {args.command}")


def _run_management_cli_command(runtime: Runtime, args: argparse.Namespace) -> None:
    if args.command == "checkpoint":
        _print_json(_run_checkpoint_command(runtime, args))
        return
    if args.command == "skills":
        _print_json(_run_skills_command(runtime, args))
        return
    if args.command == "capabilities":
        _print_json(_run_capabilities_command(runtime, args))
        return
    if args.command == "images":
        _print_json(_run_images_command(runtime, args))
        return
    if args.command == "jsonrpc":
        _print_json(_run_jsonrpc_command(runtime, args))
        return
    if args.command == "mcp":
        _print_json(_run_mcp_command(runtime, args))
        return
    if args.command == "modules":
        _print_json(_run_modules_command(runtime, args))
        return
    if args.command == "human":
        _print_json(
            [request.__dict__ for request in runtime.human.drain_terminal_queue()]
        )
        return
    raise AssertionError(f"unsupported management command: {args.command}")


def _dispatch_cli_command(
    runtime: Runtime,
    args: argparse.Namespace,
    selected_db_display: str,
) -> None:
    if args.command in _COMPACT_COMMANDS:
        _print_json(to_jsonable(_run_compact_command(runtime, args)))
        return
    if args.command in _PRIMARY_COMMANDS:
        _run_primary_cli_command(runtime, args, selected_db_display)
        return
    if args.command in _PROCESS_COMMANDS:
        _run_process_cli_command(runtime, args)
        return
    if args.command in _MANAGEMENT_COMMANDS:
        _run_management_cli_command(runtime, args)
        return
    raise AssertionError(f"unsupported CLI command: {args.command}")


def _dispatch_cli_command_with_shutdown(
    runtime: Runtime,
    args: argparse.Namespace,
    selected_db_display: str,
) -> None:
    command_error: BaseException | None = None
    try:
        _dispatch_cli_command(runtime, args, selected_db_display)
    except BaseException as exc:
        command_error = exc
        raise
    finally:
        try:
            _shutdown_runtime_before_exit(runtime)
        except BaseException as shutdown_error:
            if command_error is None:
                raise
            command_error.add_note(
                "CLI runtime teardown also failed: "
                f"{type(shutdown_error).__name__}: {shutdown_error}"
            )


def main(argv: list[str] | None = None) -> None:
    parser, args = _parse_cli_args(argv)
    selected_config = _load_runtime_config(args.config, parser)
    if args.command == "modules" and args.modules_command == "verify":
        return _print_json(_verify_module_manifest(selected_config, args))
    try:
        selected_db_display = display_store_target(args.db, config=selected_config)
    except LibOSValidationError as exc:
        parser.error(str(exc))
    runtime = Runtime.open(
        args.db,
        config=selected_config,
        module_manifests=args.module_manifest,
        trusted_modules=args.trusted_module,
        trusted_module_sha256=args.trusted_module_sha256,
    )
    _dispatch_cli_command_with_shutdown(runtime, args, selected_db_display)


def cli(argv: list[str] | None = None) -> None:
    """Console boundary that maps domain errors to stable JSON and exit codes."""

    try:
        main(argv)
    except LibOSError as exc:
        _print_json(
            {
                "schema_version": 1,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        )
        raise SystemExit(1) from None


def _add_spawn_authority_manifest_parser_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--authority-manifest-json",
        help=(
            "Host-authored TaskAuthorityManifest template as a JSON object. "
            "Omit to use an implicit launch manifest; an explicit '{}' is an empty authority ceiling."
        ),
    )


def _verify_module_manifest(config: AgentLibOSConfig, args: argparse.Namespace) -> dict[str, Any]:
    return ModuleLoader(
        config,
        trusted_modules=tuple(args.trusted_module),
        trusted_sha256=tuple(args.trusted_module_sha256),
    ).verify(args.path)


def _resource_summary(runtime: Runtime, pid: str) -> dict[str, Any]:
    process = runtime.process.get(pid)
    return {
        "pid": pid,
        "status_message": process.status_message,
        **process_state_to_mapping(
            process.status.value,
            process.wait_state,
            process.outcome,
            process.state_generation,
        ),
        "budget": to_jsonable(process.resource_budget),
        "usage": to_jsonable(process.resource_usage),
        "remaining": to_jsonable(runtime.resources.remaining_budget(pid)),
    }


def _add_explain_parser_args(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="explain_command", required=True)
    process = sub.add_parser("process", help="List explainable operations for one process")
    process.add_argument("pid")
    process.add_argument("--limit", type=int)
    process.add_argument("--cursor")
    operation = sub.add_parser("operation", help="Explain one operation and its causal tree")
    operation.add_argument("operation_id")
    operation.add_argument("--evidence-limit", type=int)
    operation.add_argument("--cursor")
    for kind in ("call", "effect", "request", "audit", "event", "reservation", "context"):
        evidence = sub.add_parser(kind, help=f"Resolve {kind} evidence to its operation")
        evidence.add_argument("evidence_id")
        evidence.add_argument("--evidence-limit", type=int)
        evidence.add_argument("--cursor")


def _run_explain_command(runtime: Runtime, args: argparse.Namespace) -> dict[str, Any]:
    command = args.explain_command
    if command == "process":
        return runtime.explain.list_operations(args.pid, limit=args.limit, cursor=args.cursor)
    if command == "operation":
        return runtime.explain.explain_operation(
            args.operation_id,
            evidence_limit=args.evidence_limit,
            cursor=args.cursor,
        )
    if command in {"call", "effect", "request", "audit", "event", "reservation", "context"}:
        return runtime.explain.resolve(
            command,
            args.evidence_id,
            evidence_limit=args.evidence_limit,
            cursor=args.cursor,
        )
    raise SystemExit(f"unknown explain command: {command}")


def _add_workflow_parser_args(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="workflow_command", required=True)
    run_parser = sub.add_parser("run", help=_WORKFLOW_RUN_HELP)
    run_parser.add_argument("tool", help="Tool/workflow name to run.")
    run_parser.add_argument("--args-json", default="{}", help="JSON object passed as tool arguments.")
    run_parser.add_argument("--image", help="AgentImage id to use; defaults to the runtime default image.")
    run_parser.add_argument("--goal", help="Optional process goal; defaults to workflow:<tool>.")
    run_parser.add_argument("--working-directory", help="Optional AgentProcess working directory.")
    run_parser.add_argument(
        "--authority-manifest-json",
        help=(
            "Host/workflow-controller TaskAuthorityManifest template as a JSON object. "
            "Omit to permit controller-derived launch authority; an explicit '{}' suppresses it."
        ),
    )


def _run_workflow_command(runtime: Runtime, args: argparse.Namespace) -> Any:
    if args.workflow_command != "run":
        raise SystemExit(f"unknown workflow command: {args.workflow_command}")
    return runtime.run_workflow(
        args.tool,
        _parse_json_mapping(args.args_json, "--args-json"),
        image=args.image,
        goal=args.goal,
        working_directory=args.working_directory,
        authority_manifest=_parse_optional_json_mapping(
            args.authority_manifest_json,
            "--authority-manifest-json",
        ),
    )


def _add_task_run_parser_args(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="task_run_command", required=True)

    start = sub.add_parser(
        "start",
        help="Create a queued durable TaskRun; use --run to consume scheduler quanta now",
    )
    goal = start.add_mutually_exclusive_group(required=True)
    goal.add_argument("--goal", help="Task goal as text.")
    goal.add_argument("--goal-json", help="Task goal as strict JSON.")
    start.add_argument(
        "--title",
        "--display-title",
        dest="title",
        required=True,
        help="Persistent display title for the run.",
    )
    start.add_argument(
        "--image",
        "--image-id",
        dest="image",
        help="AgentImage id; omitted uses the Runtime's configured default image.",
    )
    start.add_argument(
        "--launch-json",
        "--launch-options-json",
        dest="launch_json",
        default="{}",
        help="Host-authored launch options as a strict JSON object.",
    )
    start.add_argument(
        "--authority-manifest-id",
        "--authority-ref",
        dest="authority_manifest_id",
        help="Reference to an existing Host-authored authority manifest.",
    )
    start.add_argument(
        "--deadline-at",
        help="Optional absolute wall-clock deadline timestamp.",
    )
    start.add_argument(
        "--retention",
        choices=_TASK_RUN_RETENTIONS,
        default="purge_on_terminal",
        help="Durable payload retention policy; permanent is Host/admin only.",
    )
    start.add_argument(
        "--client-request-id",
        help="Stable client request id for create idempotency; omitted generates one.",
    )
    start.add_argument(
        "--run",
        action="store_true",
        help="After creation, run until the run blocks, finishes, or exhausts --max-quanta.",
    )
    start.add_argument(
        "--run-command-id",
        help="Stable command id for the separate --run mutation.",
    )
    start.add_argument(
        "--max-quanta",
        type=int,
        help="Optional scheduler quantum budget used only with --run.",
    )

    get = sub.add_parser("get", help="Get the current durable TaskRun summary")
    get.add_argument("run_id")

    list_parser = sub.add_parser("list", help="List durable TaskRuns as a keyset-paginated page")
    list_parser.add_argument(
        "--status",
        dest="statuses",
        action="append",
        default=[],
        help="Filter status, comma-separated or repeated.",
    )
    list_parser.add_argument("--cursor", help="Opaque keyset cursor from the preceding page.")
    list_parser.add_argument(
        "--limit",
        type=_task_run_page_limit,
        default=100,
        help="Page size from 1 through 500.",
    )

    wait = sub.add_parser(
        "wait",
        help="Observe a TaskRun until its revision changes, it blocks, or it is terminal",
    )
    wait.add_argument("run_id")
    wait.add_argument("--timeout", type=float, help="Optional finite wait timeout in seconds.")

    recovery_options = sub.add_parser(
        "recovery-options",
        help="List the server-derived recovery options currently allowed for a TaskRun",
    )
    recovery_options.add_argument("run_id")

    pause = sub.add_parser("pause", help="Durably pause a TaskRun before new dispatch")
    pause.add_argument("run_id")
    _add_task_run_command_args(pause)

    resume = sub.add_parser("resume", help="Resume a paused TaskRun")
    resume.add_argument("run_id")
    _add_task_run_command_args(resume)

    cancel = sub.add_parser("cancel", help="Cancel a TaskRun with explicit confirmation")
    cancel.add_argument("run_id")
    cancel.add_argument("--reason")
    cancel.add_argument(
        "--confirm",
        action="store_true",
        required=True,
        help="Confirm durable cancellation and descendant termination.",
    )
    _add_task_run_command_args(cancel)

    follow_up = sub.add_parser(
        "follow-up",
        help="Append a durable requirement and notify the current root Agent",
    )
    follow_up.add_argument("run_id")
    follow_up.add_argument(
        "content",
        nargs="?",
        help="Follow-up requirement text; alternatively use --content or --content-json.",
    )
    follow_up.add_argument("--content", dest="content_option", help="Follow-up requirement text.")
    follow_up.add_argument("--content-json", help="Follow-up requirement as strict JSON.")
    follow_up.add_argument(
        "--interrupt",
        action="store_true",
        help="Also interrupt the active root process after durably recording the requirement.",
    )
    follow_up.add_argument(
        "--optional",
        action="store_false",
        dest="required",
        default=True,
        help="Record the follow-up without making it a required success condition.",
    )
    _add_task_run_command_args(follow_up)

    recover = sub.add_parser(
        "recover",
        help="Apply a server-provided recovery option with explicit confirmation",
    )
    recover.add_argument("run_id")
    recover.add_argument(
        "option",
        nargs="?",
        help="Exact recovery option returned by the server; alternatively use --option.",
    )
    recover.add_argument("--option", dest="option_flag", help="Exact server-provided recovery option.")
    recover.add_argument(
        "--receipt-json",
        "--evidence-json",
        dest="receipt_json",
        help="Optional authoritative receipt as a strict JSON object.",
    )
    recover.add_argument(
        "--confirm",
        action="store_true",
        required=True,
        help="Confirm the selected evidence-backed recovery action.",
    )
    _add_task_run_command_args(recover)

    rerun = sub.add_parser("rerun", help="Create a new TaskRun linked to a terminal source run")
    rerun.add_argument("run_id")
    rerun.add_argument("--client-request-id", help="Stable create id for the linked rerun.")
    rerun.add_argument(
        "--spec-overrides-json",
        help="Optional complete or partial canonical TaskRun spec overrides.",
    )
    _add_task_run_command_args(rerun)


def _add_task_run_command_args(
    parser: argparse.ArgumentParser,
    *,
    expected_revision: bool = True,
) -> None:
    if expected_revision:
        parser.add_argument(
            "--expected-revision",
            type=_task_run_revision_arg,
            required=True,
            help="Revision last observed by this client; stale revisions are rejected.",
        )
    parser.add_argument(
        "--command-id",
        help="Stable idempotency id; omitted generates one for this invocation.",
    )


def _task_run_revision_arg(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("revision must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("revision must be a non-negative integer")
    return parsed


def _task_run_page_limit(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("task-run list limit must be an integer from 1 through 500") from exc
    if not 1 <= parsed <= 500:
        raise argparse.ArgumentTypeError("task-run list limit must be an integer from 1 through 500")
    return parsed


def _run_task_run_command(runtime: Runtime, args: argparse.Namespace) -> Any:
    manager = runtime.task_runs
    command = args.task_run_command
    if command == "start":
        return _run_task_run_start_command(runtime, args)
    if command in {"get", "list", "wait", "recovery-options"}:
        return _run_task_run_read_command(manager, args)

    mutation = {
        "expected_revision": args.expected_revision,
        "command_id": _task_run_command_id(args.command_id),
    }
    if command == "pause":
        return to_jsonable(manager.pause(args.run_id, **mutation))
    if command == "resume":
        return to_jsonable(manager.resume(args.run_id, **mutation))
    if command == "cancel":
        return to_jsonable(
            manager.cancel(
                args.run_id,
                **mutation,
                reason=args.reason or "",
            )
        )
    if command == "follow-up":
        return to_jsonable(
            manager.follow_up(
                args.run_id,
                **mutation,
                body=_task_run_follow_up_content(runtime, args),
                kind="interrupt" if args.interrupt else "normal",
                required=args.required,
            )
        )
    if command == "recover":
        return to_jsonable(
            manager.recover(
                args.run_id,
                **mutation,
                option_id=_task_run_recovery_option(args),
                receipt=(
                    _task_run_json_mapping(runtime, args.receipt_json, "--receipt-json")
                    if args.receipt_json is not None
                    else None
                ),
            )
        )
    if command == "rerun":
        return to_jsonable(
            manager.rerun(
                args.run_id,
                **mutation,
                client_request_id=(
                    _task_run_client_request_id(args.client_request_id)
                    if args.client_request_id is not None
                    else None
                ),
                spec_overrides=(
                    _task_run_json_mapping(
                        runtime,
                        args.spec_overrides_json,
                        "--spec-overrides-json",
                    )
                    if args.spec_overrides_json is not None
                    else None
                ),
            )
        )
    raise SystemExit(f"unknown task-run command: {command}")


def _run_task_run_start_command(runtime: Runtime, args: argparse.Namespace) -> Any:
    if args.run_command_id is not None and not args.run:
        raise LibOSValidationError("--run-command-id requires --run")
    if args.max_quanta is not None and not args.run:
        raise LibOSValidationError("--max-quanta requires --run")
    if args.max_quanta is not None and args.max_quanta <= 0:
        raise LibOSValidationError("TaskRun max_quanta must be a positive integer")

    manager = runtime.task_runs
    created = manager.create(
        _task_run_spec_from_args(runtime, args),
        client_request_id=_task_run_client_request_id(args.client_request_id),
        auto_run=False,
    )
    if not args.run:
        return to_jsonable(created)
    return to_jsonable(
        manager.run_until_blocked(
            _task_run_summary_field(created, "run_id", str),
            expected_revision=_task_run_summary_field(created, "revision", int),
            command_id=_task_run_command_id(args.run_command_id),
            max_quanta=args.max_quanta,
        )
    )


def _run_task_run_read_command(manager: Any, args: argparse.Namespace) -> Any:
    command = args.task_run_command
    if command == "get":
        return to_jsonable(manager.get(args.run_id))
    if command == "list":
        return to_jsonable(
            manager.list(
                statuses=_task_run_status_filters(args.statuses),
                cursor=args.cursor,
                limit=args.limit,
            )
        )
    if command == "wait":
        return to_jsonable(
            manager.wait(
                args.run_id,
                timeout=_finite_timeout_or_none(args.timeout, "--timeout"),
            )
        )
    if command == "recovery-options":
        return to_jsonable(manager.recovery_options(args.run_id))
    raise SystemExit(f"unknown task-run read command: {command}")


def _task_run_spec_from_args(runtime: Runtime, args: argparse.Namespace) -> Any:
    # Keep the import local so the ordinary CLI remains importable in runtimes
    # that reject the v4 schema before TaskRun services are assembled.
    from agent_libos.models import TaskRunSpecV1

    goal = (
        args.goal
        if args.goal is not None
        else _task_run_json_value(runtime, args.goal_json, "--goal-json")
    )
    values = {
        "schema_version": 1,
        "goal": goal,
        "display_title": args.title,
        "image_id": args.image,
        "launch_options": _task_run_json_mapping(runtime, args.launch_json, "--launch-json"),
        "authority_manifest_id": args.authority_manifest_id,
        "deadline_at": args.deadline_at,
        "retention": args.retention,
    }
    try:
        from_mapping = getattr(TaskRunSpecV1, "from_mapping", None)
        if callable(from_mapping):
            return from_mapping(values)
        return TaskRunSpecV1(**values)
    except (TypeError, ValueError) as exc:
        raise LibOSValidationError(str(exc)) from exc


def _task_run_json_value(runtime: Runtime, value: str, label: str) -> Any:
    max_bytes = runtime.config.task_runs.payload_max_bytes
    try:
        return bounded_json_loads(
            value,
            max_bytes=max_bytes,
        )
    except ValueError as exc:
        raise LibOSValidationError(f"{label} must be valid bounded JSON: {exc}") from exc


def _task_run_json_mapping(runtime: Runtime, value: str, label: str) -> dict[str, Any]:
    decoded = _task_run_json_value(runtime, value, label)
    if not isinstance(decoded, dict):
        raise LibOSValidationError(f"{label} must be a JSON object")
    return decoded


def _task_run_command_id(value: str | None) -> str:
    if value is None:
        return str(uuid.uuid4())
    selected = value.strip()
    if not selected:
        raise LibOSValidationError("--command-id must be a non-empty string")
    return selected


def _task_run_client_request_id(value: str | None) -> str:
    if value is None:
        return str(uuid.uuid4())
    selected = value.strip()
    if not selected:
        raise LibOSValidationError("--client-request-id must be a non-empty string")
    return selected


def _optional_nonempty_cli_value(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    selected = value.strip()
    if not selected:
        raise LibOSValidationError(f"{label} must be a non-empty string")
    return selected


def _task_run_summary_field(summary: Any, field: str, expected_type: type[Any]) -> Any:
    projected = to_jsonable(summary)
    if not isinstance(projected, dict) or field not in projected:
        raise LibOSValidationError(f"TaskRun create returned no {field}")
    value = projected[field]
    if expected_type is int:
        valid = isinstance(value, int) and not isinstance(value, bool) and value >= 0
    else:
        valid = isinstance(value, expected_type) and bool(str(value).strip())
    if not valid:
        raise LibOSValidationError(f"TaskRun create returned an invalid {field}")
    return value


def _task_run_status_filters(values: list[str]) -> tuple[str, ...] | None:
    selected = _parse_csv_values(values)
    invalid = sorted(set(selected) - _TASK_RUN_STATUSES)
    if invalid:
        raise LibOSValidationError(
            "invalid TaskRun status filter(s): " + ", ".join(invalid)
        )
    return tuple(selected) or None


def _task_run_follow_up_content(runtime: Runtime, args: argparse.Namespace) -> str:
    supplied = [
        args.content is not None,
        args.content_option is not None,
        args.content_json is not None,
    ]
    if sum(supplied) != 1:
        raise LibOSValidationError(
            "follow-up requires exactly one of positional content, --content, or --content-json"
        )
    if args.content_json is not None:
        decoded = _task_run_json_value(runtime, args.content_json, "--content-json")
        return json.dumps(
            decoded,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    selected = args.content if args.content is not None else args.content_option
    assert selected is not None
    if not selected.strip():
        raise LibOSValidationError("follow-up content must be non-empty text")
    return selected


def _task_run_recovery_option(args: argparse.Namespace) -> str:
    if (args.option is None) == (args.option_flag is None):
        raise LibOSValidationError(
            "recover requires exactly one of positional option or --option"
        )
    value = args.option if args.option is not None else args.option_flag
    assert value is not None
    selected = value.strip()
    if not selected:
        raise LibOSValidationError("recovery option must be a non-empty string")
    return selected


def _add_object_task_parser_args(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="object_task_command", required=True)
    start = sub.add_parser("start", help="Start an Object-bound tool task")
    start.add_argument("--pid", required=True, help="Creator process id.")
    owner = start.add_mutually_exclusive_group(required=True)
    owner.add_argument("--owner-oid", help="Owner object id.")
    owner.add_argument("--owner-name", help="Owner object name in the selected namespace.")
    start.add_argument("--namespace", help="Namespace for --owner-name.")
    start.add_argument("tool", help="Tool bound in the creator's complete process tool table.")
    start.add_argument("--args-json", default="{}", help="JSON object passed as tool arguments.")
    start.add_argument("--notify-pid", help="Process to notify; defaults to --pid.")
    start.add_argument(
        "--notify-kind",
        choices=[kind.value for kind in ProcessMessageKind],
        default=ProcessMessageKind.NORMAL.value,
    )
    start.add_argument("--notify-channel", help="Process-message channel; defaults to object-task.")
    start.add_argument(
        "--watch-owner",
        action="store_true",
        help="Notify the runner process when the owner object is updated or linked.",
    )
    start.add_argument(
        "--watch-events",
        action="append",
        default=[],
        help="Owner events to watch, comma-separated or repeated. Defaults to updated,linked when --watch-owner is set.",
    )
    start.add_argument("--watch-channel", help="Runner message channel for owner-watch notices; defaults to object-task-owner.")
    start.add_argument(
        "--watch-kind",
        choices=[kind.value for kind in ProcessMessageKind],
        default=ProcessMessageKind.NORMAL.value,
    )
    start.add_argument(
        "--grant-result-to-notify",
        action="store_true",
        help="Try to grant result read authority to notify pid; requires object grant authority.",
    )
    start.add_argument(
        "--wait",
        action="store_true",
        required=True,
        help=(
            "Required by the one-shot CLI; wait until the task reaches a terminal "
            "or explicit waiting state before printing."
        ),
    )
    start.add_argument("--timeout", type=float, help="Optional wait timeout in seconds.")

    get = sub.add_parser("get", help="Inspect an Object task")
    get.add_argument("task_id")
    get.add_argument("--pid", help="Actor process id for visibility checks.")

    list_parser = sub.add_parser("list", help="List Object tasks")
    list_parser.add_argument("--pid", help="Actor process id for visibility checks.")
    list_parser.add_argument("--owner-oid", help="Filter by owner object id.")
    list_parser.add_argument("--active", action="store_true", help="Only include non-terminal tasks.")
    list_parser.add_argument("--limit", type=int)

    cancel = sub.add_parser("cancel", help="Cancel an Object task")
    cancel.add_argument("task_id")
    cancel.add_argument("--pid", required=True, help="Actor process id.")
    cancel.add_argument("--reason")

    wait = sub.add_parser("wait", help="Wait for an Object task to finish or enter an explicit waiting state")
    wait.add_argument("task_id")
    wait.add_argument("--pid", help="Actor process id for visibility checks.")
    wait.add_argument("--timeout", type=float)

    watch = sub.add_parser("watch-owner", help="Enable, disable, or update owner-change notices for an Object task")
    watch.add_argument("task_id")
    watch.add_argument("--pid", required=True, help="Actor process id.")
    watch.add_argument("--disable", action="store_true", help="Disable owner-change notices.")
    watch.add_argument(
        "--watch-events",
        action="append",
        default=[],
        help="Owner events to watch, comma-separated or repeated.",
    )
    watch.add_argument("--watch-channel", help="Runner message channel for owner-watch notices.")
    watch.add_argument(
        "--watch-kind",
        choices=[kind.value for kind in ProcessMessageKind],
        default=None,
    )


def _run_object_task_command(runtime: Runtime, args: argparse.Namespace) -> Any:
    command = args.object_task_command
    if command == "start":
        if not args.wait:
            raise SystemExit(
                "object-task start requires --wait in the one-shot CLI; use the GUI server or an embedded Runtime for detached ObjectTask supervision"
            )
        owner = _object_task_owner_handle(runtime, args.pid, args.owner_oid, args.owner_name, args.namespace)
        task = runtime.object_tasks.start(
            args.pid,
            owner,
            args.tool,
            _parse_json_mapping(args.args_json, "--args-json"),
            notify_pid=args.notify_pid,
            notify_kind=args.notify_kind,
            notify_channel=args.notify_channel,
            grant_result_to_notify=args.grant_result_to_notify,
            owner_watch=_object_task_owner_watch_args(args),
        )
        task = runtime.object_tasks.wait(task.task_id, actor_pid=args.pid, timeout=_finite_timeout_or_none(args.timeout, "--timeout"))
        return to_jsonable(task)
    if command == "get":
        return to_jsonable(runtime.object_tasks.get(args.task_id, actor_pid=args.pid))
    if command == "list":
        return to_jsonable(
            runtime.object_tasks.list(
                actor_pid=args.pid,
                owner_oid=args.owner_oid,
                include_terminal=not args.active,
                limit=args.limit,
            )
        )
    if command == "cancel":
        return to_jsonable(runtime.object_tasks.cancel(args.task_id, actor_pid=args.pid, reason=args.reason))
    if command == "wait":
        return to_jsonable(runtime.object_tasks.wait(args.task_id, actor_pid=args.pid, timeout=_finite_timeout_or_none(args.timeout, "--timeout")))
    if command == "watch-owner":
        events = _parse_csv_values(args.watch_events)
        return to_jsonable(
            runtime.object_tasks.watch_owner(
                args.task_id,
                actor_pid=args.pid,
                enabled=not args.disable,
                events=events or None,
                channel=args.watch_channel,
                kind=args.watch_kind,
            )
        )
    raise SystemExit(f"unknown object-task command: {command}")


def _object_task_owner_handle(
    runtime: Runtime,
    pid: str,
    owner_oid: str | None,
    owner_name: str | None,
    namespace: str | None,
) -> ObjectHandle:
    if owner_oid:
        return runtime.memory.handle_for_oid(
            pid,
            owner_oid,
            required_rights={ObjectRight.READ.value, ObjectRight.WRITE.value, ObjectRight.LINK.value},
        )
    if owner_name:
        return runtime.memory.handle_for_name(
            pid,
            owner_name,
            rights={ObjectRight.READ.value, ObjectRight.WRITE.value, ObjectRight.LINK.value},
            namespace=namespace,
        )
    raise SystemExit("either --owner-oid or --owner-name is required")


def _object_task_owner_watch_args(args: argparse.Namespace) -> dict[str, Any] | bool:
    events = _parse_csv_values(args.watch_events)
    enabled = bool(args.watch_owner or events or args.watch_channel or args.watch_kind != ProcessMessageKind.NORMAL.value)
    if not enabled:
        return False
    selected: dict[str, Any] = {"enabled": True, "kind": args.watch_kind}
    if events:
        selected["events"] = events
    if args.watch_channel:
        selected["channel"] = args.watch_channel
    return selected


def _parse_csv_values(values: list[str]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                parsed.append(item)
    return parsed


def _finite_timeout_or_none(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise SystemExit(f"{label} must be a finite non-negative number")
    return parsed


def _run_cd_command(runtime: Runtime, args: argparse.Namespace) -> dict[str, Any]:
    before = runtime.process.working_directory(args.pid)
    process = runtime.set_process_working_directory(args.pid, args.path)
    return {
        "pid": process.pid,
        "previous_working_directory": before,
        "working_directory": process.working_directory,
    }


async def _run_exec_command(runtime: Runtime, args: argparse.Namespace) -> dict[str, Any]:
    loaded_image = (
        _load_cli_image_from_package(runtime, args.image, replace=args.replace_image)
        if _is_image_package_arg(args.image)
        else None
    )
    target_image = loaded_image["image_id"] if loaded_image is not None else args.image
    exec_args = _parse_json_mapping(args.args_json, "--args-json")
    old_process = runtime.process.get(args.pid)
    if loaded_image is not None:
        runtime.capability.grant(
            args.pid,
            runtime.image_registry.resource_for(target_image),
            [CapabilityRight.READ],
            issued_by="cli",
        )
    old_image = old_process.image_id
    old_llm_profile = old_process.llm_profile_id
    process = runtime.exec_process(
        args.pid,
        target_image,
        args=exec_args,
        goal=args.goal,
        preserve_memory=args.preserve_memory,
        preserve_capabilities=args.preserve_capabilities,
        llm_profile_id=args.llm_profile,
    )
    results: list[Any] = []
    if args.run:
        results = await runtime.arun_until_idle(max_quanta=args.max_quanta)
        process = runtime.process.get(args.pid)
    return {
        "pid": args.pid,
        "goal": args.goal,
        "image_arg": args.image,
        "loaded_image": loaded_image,
        "exec": {
            "old_image": old_image,
            "new_image": process.image_id,
            "old_llm_profile_id": old_llm_profile,
            "new_llm_profile_id": process.llm_profile_id,
            "preserve_memory": args.preserve_memory,
            "preserve_capabilities": args.preserve_capabilities,
            "args": exec_args,
        },
        "process": _process_cli_summary(process),
        "ran": args.run,
        "results": results,
    }


def _run_exit_command(runtime: Runtime, args: argparse.Namespace) -> dict[str, Any]:
    if args.payload is not None and args.result_oid is not None:
        raise SystemExit("exit accepts at most one of --payload or --result-oid")
    result_handle: ObjectHandle | None = None
    if args.result_oid is not None:
        result_handle = runtime.memory.handle_for_oid(
            args.pid,
            args.result_oid,
            required_rights={ObjectRight.READ.value},
            optional_rights={ObjectRight.MATERIALIZE.value, ObjectRight.LINK.value, ObjectRight.DIFF.value},
            issued_by="cli.exit",
        )
    elif args.payload is not None:
        result_handle = runtime.memory.create_object(
            pid=args.pid,
            object_type=ObjectType.SUMMARY,
            payload=_parse_json_value(args.payload),
            metadata=ObjectMetadata(title="CLI process final result", tags=["final", "cli"]),
        )
    result_handle = runtime.process.exit(
        args.pid,
        result=result_handle,
        failed=args.failed,
        message=args.message,
    )
    process = runtime.process.get(args.pid)
    return {
        "pid": process.pid,
        "message": args.message,
        "result_oid": result_handle.oid if result_handle is not None else None,
        **process_state_to_mapping(
            process.status.value,
            process.wait_state,
            process.outcome,
            process.state_generation,
        ),
    }


async def _run_message_command(
    runtime: Runtime,
    args: argparse.Namespace,
    *,
    fixed_kind: ProcessMessageKind | None = None,
) -> dict[str, Any]:
    kind = fixed_kind or (ProcessMessageKind.INTERRUPT if getattr(args, "interrupt", False) else ProcessMessageKind(args.kind))
    payload = _parse_json_mapping(args.payload_json, "--payload-json")
    payload.setdefault("source", "cli.message")
    message = runtime.human.send_process_message(
        args.pid,
        args.body,
        kind=kind,
        human=args.human,
        channel=args.channel,
        correlation_id=args.correlation_id,
        reply_to=args.reply_to,
        subject=args.subject,
        payload=payload,
    )
    results: list[Any] = []
    if args.run:
        results = await runtime.arun_until_idle(max_quanta=args.max_quanta, human=args.human)
    return {
        "message": _message_cli_summary(message),
        "ran": args.run,
        "results": results,
    }


async def _run_interactive_command(runtime: Runtime, args: argparse.Namespace) -> dict[str, Any]:
    target_pid = args.pid or _single_active_process_pid(runtime)
    if target_pid is None:
        raise SystemExit("run --interactive needs --pid when there is not exactly one active process")
    _redirect_human_output_to_stderr(runtime)
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    stop = threading.Event()
    _start_interactive_input_thread(asyncio.get_running_loop(), queue, stop)
    _print_interactive_help(target_pid)

    results: list[Any] = []
    posted: list[dict[str, Any]] = []
    state = {"pid": target_pid, "shown_request_id": ""}
    remaining: int | None = (
        runtime.config.runtime.run_until_idle_max_quanta
        if args.max_quanta is None
        else args.max_quanta
    )
    selected_human = args.human or runtime.config.runtime.default_human
    try:
        while remaining is None or remaining > 0:
            command = _drain_interactive_queue(runtime, queue, state, selected_human, args.message_channel, posted)
            if command in {"exit", "eof"}:
                break

            batch = await runtime.scheduler.arun_until_idle(runtime.arun_process_once, max_quanta=remaining)
            results.extend(batch)
            if remaining is not None:
                remaining -= len(batch)

            processed = _process_interactive_terminal_outputs(runtime, selected_human)
            if processed:
                runtime.audit.record(
                    actor="runtime",
                    action="runtime.human_queue_drained",
                    target=f"human:{selected_human}",
                    decision={"request_ids": [request.request_id for request in processed]},
                )

            command = _drain_interactive_queue(runtime, queue, state, selected_human, args.message_channel, posted)
            if command in {"exit", "eof"}:
                break

            target = runtime.process.get(state["pid"])
            if target.status in _TERMINAL_PROCESS_STATUSES:
                break
            _show_pending_interactive_human_request(runtime, selected_human, state)
            if batch or processed:
                await asyncio.sleep(0)
                continue
            try:
                line = await asyncio.wait_for(queue.get(), timeout=runtime.scheduler.poll_interval_s)
            except asyncio.TimeoutError:
                continue
            command = _handle_interactive_line(runtime, line, state, selected_human, args.message_channel, posted)
            if command in {"exit", "eof"}:
                break
    finally:
        stop.set()
    return {
        "interactive": True,
        "target_pid": state["pid"],
        "posted_messages": posted,
        "results": results,
        "remaining_quanta": remaining,
        "process": _process_cli_summary(runtime.process.get(state["pid"])),
    }


def _load_cli_image_from_package(runtime: Runtime, value: str, *, replace: bool) -> dict[str, Any]:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        raise SystemExit(f"image package does not exist: {path}")
    result = runtime.image_registry.register_from_package_path(
        path,
        actor="cli",
        replace=replace,
        require_capability=False,
    )
    return {
        "image_id": result.image.image_id,
        "name": result.image.name,
        "version": result.image.version,
        "replaced": result.replaced,
        "source": result.source,
        "boot": result.image.boot,
        "package_sha256": result.image.metadata.get("package_sha256"),
    }


def _is_image_package_arg(value: str) -> bool:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    manifest_name = DEFAULT_CONFIG.image.package_manifest_name
    return (
        path.is_dir() and (path / manifest_name).is_file()
    ) or (
        path.is_file() and path.name == manifest_name
    )


def _parse_json_mapping(value: str, label: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} must be valid JSON") from exc
    if not isinstance(decoded, dict):
        raise SystemExit(f"{label} must be a JSON object")
    return decoded


def _parse_optional_json_mapping(value: str | None, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return _parse_json_mapping(value, label)


def _parse_json_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {"content": value}


def _add_message_parser_args(parser: argparse.ArgumentParser, *, include_kind: bool = True) -> None:
    parser.add_argument("pid", help="Target process id.")
    parser.add_argument("body", help="Message body. Quote it to include spaces.")
    if include_kind:
        parser.add_argument("--kind", choices=[kind.value for kind in ProcessMessageKind], default=ProcessMessageKind.NORMAL.value)
        parser.add_argument("--interrupt", action="store_true", help="Shortcut for --kind interrupt.")
    parser.add_argument("--human", help="Human actor name.")
    parser.add_argument("--channel", default="human", help="Process-message channel.")
    parser.add_argument("--subject", help="Short message subject.")
    parser.add_argument("--correlation-id", help="Optional conversation/request correlation id.")
    parser.add_argument("--reply-to", help="Optional message id this message replies to.")
    parser.add_argument("--payload-json", default="{}", help="Structured JSON object to include in the message payload.")
    parser.add_argument("--run", action="store_true", help="Run the scheduler after posting the message.")
    parser.add_argument(
        "--max-quanta",
        type=int,
        help=(
            "Optional quantum budget when --run is set; omitted uses "
            "runtime.run_until_idle_max_quanta (which may be unbounded)."
        ),
    )


def _add_checkpoint_parser_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--actor-pid",
        help="If set, execute as this process and enforce checkpoint capabilities. Omit for admin CLI mode.",
    )
    sub = parser.add_subparsers(dest="checkpoint_command", required=True)
    create = sub.add_parser("create", help="Create a checkpoint for a process subtree")
    create.add_argument("pid")
    create.add_argument("reason")
    create.add_argument("--metadata-json", default="{}", help="Optional checkpoint metadata JSON object.")
    list_parser = sub.add_parser("list", help="List checkpoints")
    list_parser.add_argument("--pid", help="Filter by process id.")
    list_parser.add_argument("--limit", type=int)
    inspect = sub.add_parser("inspect", help="Inspect checkpoint metadata")
    inspect.add_argument("checkpoint_id")
    diff = sub.add_parser("diff", help="Diff current reconstructable state against a checkpoint")
    diff.add_argument("checkpoint_id")
    restore = sub.add_parser("restore", help="Restore checkpoint process subtree")
    restore.add_argument("checkpoint_id")
    fork = sub.add_parser("fork", help="Fork a new process subtree from a checkpoint")
    fork.add_argument("checkpoint_id")
    fork.add_argument("--parent-pid", help="Optional parent pid for the fork root.")
    replay = sub.add_parser("replay", help="Return diagnostic event timeline from a checkpoint to an event")
    replay.add_argument("checkpoint_id")
    replay.add_argument("event_id")


def _run_checkpoint_command(runtime: Runtime, args: argparse.Namespace) -> dict[str, Any] | list[dict[str, Any]]:
    actor = args.actor_pid or "cli"
    require_capability = args.actor_pid is not None
    command = args.checkpoint_command
    if command == "create":
        checkpoint_id = runtime.checkpoint.create(
            args.pid,
            args.reason,
            actor=actor,
            require_capability=require_capability,
            metadata=_parse_json_mapping(args.metadata_json, "--metadata-json"),
        )
        return {"checkpoint_id": checkpoint_id, "pid": args.pid, "reason": args.reason, "actor": actor}
    if command == "list":
        return runtime.checkpoint.list(
            args.pid,
            actor=actor if require_capability else None,
            require_capability=require_capability,
            limit=args.limit,
        )
    if command == "inspect":
        return runtime.checkpoint.inspect(
            args.checkpoint_id,
            actor=actor if require_capability else None,
            require_capability=require_capability,
        )
    if command == "diff":
        return runtime.checkpoint.diff(
            args.checkpoint_id,
            actor=actor if require_capability else None,
            require_capability=require_capability,
        )
    if command == "restore":
        return runtime.checkpoint.restore(actor, args.checkpoint_id, require_capability=require_capability)
    if command == "fork":
        return runtime.checkpoint.fork_from_checkpoint(
            actor,
            args.checkpoint_id,
            parent_pid=args.parent_pid,
            require_capability=require_capability,
        )
    if command == "replay":
        return runtime.checkpoint.replay_to_event(
            args.checkpoint_id,
            args.event_id,
            actor=actor if require_capability else "cli",
            require_capability=require_capability,
        )
    raise SystemExit(f"unknown checkpoint command: {command}")


def _parse_exact_sha256(value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise argparse.ArgumentTypeError(
            "must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def _add_skills_parser_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--actor-pid",
        help=(
            "If set, execute supported subcommands as this process and enforce Skill capabilities. "
            "The Host-filesystem 'validate' subcommand rejects this option. Omit for admin CLI mode."
        ),
    )
    sub = parser.add_subparsers(dest="skills_command", required=True)
    discover = sub.add_parser("discover", help="Discover registered skills")
    discover.add_argument("--text")
    discover.add_argument("--limit", type=int)
    inspect = sub.add_parser("inspect", help="Inspect a registered skill package")
    inspect.add_argument("skill_id")
    validate = sub.add_parser("validate", help="Validate a standard Agent Skill directory or SKILL.md")
    validate.add_argument("path")
    register = sub.add_parser("register", help="Register a standard Agent Skill directory or SKILL.md")
    register.add_argument("path")
    register.add_argument("--replace", action="store_true")
    register.add_argument("--source-type", choices=["workspace", "global", "runtime"], default=None)
    activate = sub.add_parser("activate", help="Activate a registered skill in a process")
    activate.add_argument("pid")
    activate.add_argument("skill_id")
    activate.add_argument(
        "--expected-package-sha256",
        required=True,
        type=_parse_exact_sha256,
        help=(
            "Exact package_sha256 returned by skills discover for the package "
            "version to activate."
        ),
    )
    unload = sub.add_parser("unload", help="Unload a skill from a process")
    unload.add_argument("pid")
    unload.add_argument("skill_id")
    trust = sub.add_parser("trust", help="Trust a global Skill package by exact package SHA256")
    trust.add_argument("path")
    untrust = sub.add_parser("untrust", help="Remove trust for the current bytes of a global Skill package")
    untrust.add_argument("path")


def _run_skills_command(runtime: Runtime, args: argparse.Namespace) -> dict[str, Any] | list[dict[str, Any]]:
    actor = args.actor_pid or "cli"
    require_capability = args.actor_pid is not None
    command = args.skills_command
    if command == "discover":
        return runtime.skills.discover_skills(
            args.text,
            actor=actor if require_capability else None,
            require_capability=require_capability,
            limit=args.limit,
        )
    if command == "inspect":
        return runtime.skills.inspect_skill(
            args.skill_id,
            actor=actor if require_capability else None,
            require_capability=require_capability,
        )
    if command == "validate":
        if require_capability:
            raise SystemExit(
                "skills validate is a Host filesystem operation and does not support --actor-pid; "
                "use skills register --actor-pid to validate and register through process filesystem authority"
            )
        return runtime.skills.validate_package_path(args.path)
    if command == "register":
        if args.source_type == "global":
            return runtime.skills.register_global_skill_from_path(
                args.path,
                actor=actor,
                replace=args.replace,
                require_capability=require_capability,
            )
        if require_capability:
            return runtime.skills.register_skill_from_workspace_path(
                actor,
                args.path,
                replace=args.replace,
                require_capability=True,
            )
        return runtime.skills.register_skill_from_path(
            args.path,
            actor=actor,
            replace=args.replace,
            require_capability=require_capability,
            source_type=args.source_type,
        )
    if command == "activate":
        return runtime.skills.activate_skill(
            args.pid,
            args.skill_id,
            actor=actor,
            require_capability=require_capability,
            expected_package_sha256=args.expected_package_sha256,
        )
    if command == "unload":
        return runtime.skills.unload_skill(args.pid, args.skill_id, actor=actor, require_capability=require_capability)
    if command == "trust":
        info = runtime.skills.global_package_info(args.path)
        return runtime.skills.trust_skill_source(
            actor=actor,
            source_type="global",
            source=info["source"],
            package_sha256=info["package_sha256"],
            require_capability=require_capability,
            metadata={"path": info["path"]},
        )
    if command == "untrust":
        info = runtime.skills.global_package_info(args.path)
        return runtime.skills.untrust_skill_source(
            actor=actor,
            source_type="global",
            source=info["source"],
            package_sha256=info["package_sha256"],
            require_capability=require_capability,
        )
    raise SystemExit(f"unknown skills command: {command}")


def _add_capabilities_parser_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--actor-pid",
        help="If set, execute as this process and enforce capability authority. Omit for admin CLI mode.",
    )
    sub = parser.add_subparsers(dest="capabilities_command", required=True)
    list_parser = sub.add_parser("list", help="List capabilities")
    list_parser.add_argument("--subject", help="Subject pid/name to list. Defaults to actor pid in process mode; all in admin mode.")
    list_parser.add_argument("--include-inactive", action="store_true")
    list_parser.add_argument("--limit", type=int)
    inspect = sub.add_parser("inspect", help="Inspect one capability")
    inspect.add_argument("capability_id")
    grant = sub.add_parser("grant", help="Issue a new capability")
    grant.add_argument("subject")
    _add_capability_spec_args(grant)
    delegate = sub.add_parser("delegate", help="Delegate an attenuated capability from parent to child")
    delegate.add_argument("parent")
    delegate.add_argument("child")
    _add_capability_spec_args(delegate)
    revoke = sub.add_parser("revoke", help="Revoke a capability")
    revoke.add_argument("capability_id")
    revoke.add_argument("--reason")
    explain = sub.add_parser("explain", help="Explain an authorization decision")
    explain.add_argument("subject")
    explain.add_argument("resource")
    explain.add_argument("right", choices=[right.value for right in CapabilityRight])
    explain.add_argument("--context-json", default="{}", help="Optional authorization context JSON object.")


def _add_capability_spec_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("resource")
    parser.add_argument("--rights", nargs="+", required=True, choices=[right.value for right in CapabilityRight])
    parser.add_argument("--effect", choices=[effect.value for effect in CapabilityEffect], default=CapabilityEffect.ALLOW.value)
    parser.add_argument("--delegable", action="store_true")
    parser.add_argument("--revocable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--uses-remaining", type=int)
    parser.add_argument("--expires-at")
    parser.add_argument("--constraints-json", default="{}", help="Capability constraint JSON object.")
    parser.add_argument("--metadata-json", default="{}", help="Capability metadata JSON object.")


def _validated_capability_list_limit(
    runtime: Runtime,
    value: object,
) -> int:
    configured_limit = runtime.config.capability.list_limit
    selected = configured_limit if value is None else value
    if type(selected) is not int or selected <= 0:
        raise LibOSValidationError(
            "capability list limit must be a positive integer"
        )
    if selected > configured_limit:
        raise LibOSValidationError(
            f"capability list limit must be at most {configured_limit}"
        )
    return selected


def _list_capabilities_for_cli(
    runtime: Runtime,
    args: argparse.Namespace,
    *,
    actor: str,
    process_mode: bool,
) -> list[dict[str, Any]]:
    selected_limit = _validated_capability_list_limit(runtime, args.limit)
    subject = args.subject or (actor if process_mode else None)
    if process_mode and subject != actor:
        runtime.capability.require(
            actor,
            f"process:{subject}",
            CapabilityRight.ADMIN,
        )
    caps = runtime.capability.list_for_presentation(
        subject=subject,
        include_inactive=args.include_inactive,
        limit=selected_limit,
    )
    return [runtime.capability.inspect(cap.cap_id) for cap in caps]


def _run_capabilities_command(runtime: Runtime, args: argparse.Namespace) -> dict[str, Any] | list[dict[str, Any]]:
    actor = args.actor_pid or "cli.admin"
    process_mode = args.actor_pid is not None
    command = args.capabilities_command
    if command == "list":
        return _list_capabilities_for_cli(
            runtime,
            args,
            actor=actor,
            process_mode=process_mode,
        )
    if command == "inspect":
        cap = runtime.store.get_capability(args.capability_id)
        if cap is None:
            raise NotFound(f"capability not found: {args.capability_id}")
        if process_mode and cap.subject != actor:
            runtime.capability.require(actor, cap.resource, CapabilityRight.ADMIN)
        return runtime.capability.inspect(args.capability_id)
    if command == "grant":
        spec = _capability_spec_from_args(args)
        cap = runtime.capability.issue(
            actor=actor,
            subject=args.subject,
            spec=spec,
            require_authority=process_mode,
        )
        return runtime.capability.inspect(cap.cap_id)
    if command == "delegate":
        if process_mode and args.parent != actor:
            raise SystemExit("--actor-pid delegation may only delegate from that actor process")
        cap = runtime.capability.delegate(args.parent, args.child, _capability_spec_from_args(args), actor=actor)
        return runtime.capability.inspect(cap.cap_id)
    if command == "revoke":
        cap = runtime.capability.revoke(
            args.capability_id,
            revoked_by=actor,
            reason=args.reason,
            require_authority=process_mode,
        )
        return runtime.capability.inspect(cap.cap_id)
    if command == "explain":
        if process_mode and args.subject != actor:
            runtime.capability.require(actor, args.resource, CapabilityRight.ADMIN)
        return runtime.capability.explain_decision(
            args.subject,
            args.resource,
            args.right,
            _parse_json_mapping(args.context_json, "--context-json"),
        )
    raise SystemExit(f"unknown capabilities command: {command}")


def _add_images_parser_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--actor-pid",
        help=(
            "If set, execute as this process and enforce filesystem, image, and checkpoint "
            "capabilities as applicable. Omit for admin CLI mode."
        ),
    )
    sub = parser.add_subparsers(dest="images_command", required=True)
    sub.add_parser("list", help="List registered AgentImages")
    inspect = sub.add_parser("inspect", help="Inspect one AgentImage")
    inspect.add_argument("image_id")
    validate = sub.add_parser("validate", help="Validate an image package directory")
    validate.add_argument("path")
    register = sub.add_parser("register", help="Register an image package directory")
    register.add_argument("path")
    register.add_argument("--replace", action="store_true")
    commit = sub.add_parser("commit", help="Commit a checkpoint into a checkpoint-derived AgentImage")
    commit.add_argument("checkpoint_id")
    commit.add_argument("image_id")
    commit.add_argument("--name", required=True)
    commit.add_argument("--version", default="v0")
    commit.add_argument("--replace", action="store_true")
    commit.add_argument("--metadata-json", default="{}", help="Optional image metadata JSON object.")


def _run_images_command(runtime: Runtime, args: argparse.Namespace) -> dict[str, Any] | list[dict[str, Any]]:
    actor = args.actor_pid or "cli"
    require_capability = args.actor_pid is not None
    command = args.images_command
    if command == "list":
        if require_capability:
            runtime.capability.require(actor, runtime.image_registry.registry_resource(), CapabilityRight.READ)
        return runtime.image_registry.list_images()
    if command == "inspect":
        if require_capability:
            runtime.capability.require(actor, runtime.image_registry.resource_for(args.image_id), CapabilityRight.READ)
        return runtime.image_registry.inspect(args.image_id)
    if command == "validate":
        if require_capability:
            return runtime.image_registry.validate_workspace_package(actor, args.path)
        return runtime.image_registry.validate_package_path(args.path)
    if command == "register":
        if require_capability:
            result = runtime.image_registry.register_from_workspace_package(actor, args.path, replace=args.replace)
        else:
            result = runtime.image_registry.register_from_package_path(
                args.path,
                actor=actor,
                replace=args.replace,
                require_capability=False,
            )
        image = result.image
        return {
            "image_id": image.image_id,
            "name": image.name,
            "version": image.version,
            "replaced": result.replaced,
            "source": result.source,
            "boot": image.boot,
            "package_sha256": image.metadata.get("package_sha256"),
            "package_jit_tools": image.metadata.get("package_jit_tools", []),
            "required_capabilities_count": len(image.required_capabilities),
            "required_modules_count": len(image.required_modules),
        }
    if command == "commit":
        result = runtime.image_registry.commit_from_checkpoint(
            actor=actor,
            checkpoint_id=args.checkpoint_id,
            image_id=args.image_id,
            name=args.name,
            version=args.version,
            replace=args.replace,
            metadata=_parse_json_mapping(args.metadata_json, "--metadata-json"),
            require_capability=require_capability,
        )
        image = result.image
        return {
            "image_id": image.image_id,
            "name": image.name,
            "version": image.version,
            "replaced": result.replaced,
            "boot": image.boot,
            "required_capabilities_count": len(image.required_capabilities),
            "required_modules_count": len(image.required_modules),
        }
    raise SystemExit(f"unknown images command: {command}")


def _add_jsonrpc_parser_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--actor-pid",
        help="If set, execute registry operations as this process and enforce JSON-RPC endpoint capabilities.",
    )
    sub = parser.add_subparsers(dest="jsonrpc_command", required=True)
    register = sub.add_parser("register", help="Register a JSON-RPC endpoint manifest from YAML or JSON")
    register.add_argument("path")
    register.add_argument("--replace", action="store_true")
    list_parser = sub.add_parser("list", help="List registered JSON-RPC endpoint metadata")
    list_parser.add_argument("--text")
    list_parser.add_argument("--limit", type=int)
    inspect = sub.add_parser("inspect", help="Inspect one registered JSON-RPC endpoint")
    inspect.add_argument("endpoint_id")
    call = sub.add_parser("call", help="Call a registered JSON-RPC method as a process")
    call.add_argument("pid")
    call.add_argument("endpoint_id")
    call.add_argument("method_id")
    call.add_argument("--params-json", help="JSON-RPC params value. Omit for no params member.")
    unregister = sub.add_parser("unregister", help="Delete a registered JSON-RPC endpoint")
    unregister.add_argument("endpoint_id")


def _run_jsonrpc_command(runtime: Runtime, args: argparse.Namespace) -> dict[str, Any] | list[dict[str, Any]]:
    actor = args.actor_pid or "cli"
    require_capability = args.actor_pid is not None
    command = args.jsonrpc_command
    if command == "register":
        if require_capability:
            cwd = runtime.process.working_directory(actor)
            read = runtime.filesystem.read_text(
                actor,
                args.path,
                max_bytes=runtime.config.jsonrpc.manifest_max_bytes,
                cwd=cwd,
            )
            return runtime.jsonrpc.register_endpoint_from_yaml_text(
                read.content,
                actor=actor,
                replace=args.replace,
                require_capability=True,
                source=read.path,
            )
        path = Path(args.path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        if not path.exists() or not path.is_file():
            raise SystemExit(f"JSON-RPC endpoint manifest does not exist: {path}")
        return runtime.jsonrpc.register_endpoint_from_yaml_text(
            path.read_text(encoding="utf-8"),
            actor=actor,
            replace=args.replace,
            require_capability=False,
            source=str(path),
        )
    if command == "list":
        return runtime.jsonrpc.list_endpoints(
            actor=actor if require_capability else None,
            require_capability=require_capability,
            text=args.text,
            limit=args.limit,
        )
    if command == "inspect":
        return runtime.jsonrpc.inspect_endpoint(
            args.endpoint_id,
            actor=actor if require_capability else None,
            require_capability=require_capability,
            include_sensitive_fields=not require_capability,
        )
    if command == "call":
        if args.actor_pid is not None and args.actor_pid != args.pid:
            raise SystemExit("jsonrpc call --actor-pid must match the target process pid")
        params = _parse_json_value(args.params_json) if args.params_json is not None else None
        return to_jsonable(runtime.jsonrpc.call(args.pid, args.endpoint_id, args.method_id, params))
    if command == "unregister":
        return runtime.jsonrpc.unregister_endpoint(
            args.endpoint_id,
            actor=actor,
            require_capability=require_capability,
        )
    raise SystemExit(f"unknown jsonrpc command: {command}")


def _add_mcp_parser_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--actor-pid",
        help="If set, execute registry operations as this process and enforce MCP server capabilities.",
    )
    sub = parser.add_subparsers(dest="mcp_command", required=True)
    register = sub.add_parser("register", help="Register an MCP server manifest from YAML or JSON")
    register.add_argument("path")
    register.add_argument("--replace", action="store_true")
    list_parser = sub.add_parser("list", help="List registered MCP server metadata")
    list_parser.add_argument("--text")
    list_parser.add_argument("--limit", type=int)
    inspect = sub.add_parser("inspect", help="Inspect one registered MCP server")
    inspect.add_argument("server_id")
    discover = sub.add_parser(
        "discover",
        help="Negotiate and inspect one registered modern MCP server",
    )
    discover.add_argument("server_id")
    tools = sub.add_parser("tools", help="List allowed tools for one registered MCP server")
    tools.add_argument("server_id")
    tools.add_argument("--refresh", action="store_true", help="Query the live MCP server for current tool metadata.")
    call = sub.add_parser("call", help="Call a registered MCP tool as a process")
    call.add_argument("pid")
    call.add_argument("server_id")
    call.add_argument("tool_id")
    call.add_argument("--arguments-json", help="MCP tool arguments object. Omit for {}.")
    unregister = sub.add_parser("unregister", help="Delete a registered MCP server")
    unregister.add_argument("server_id")


def _run_mcp_command(runtime: Runtime, args: argparse.Namespace) -> dict[str, Any] | list[dict[str, Any]]:
    actor = args.actor_pid or "cli"
    require_capability = args.actor_pid is not None
    command = args.mcp_command
    if command == "register":
        if require_capability:
            cwd = runtime.process.working_directory(actor)
            read = runtime.filesystem.read_text(
                actor,
                args.path,
                max_bytes=runtime.config.mcp.manifest_max_bytes,
                cwd=cwd,
            )
            return runtime.mcp.register_server_from_yaml_text(
                read.content,
                actor=actor,
                replace=args.replace,
                require_capability=True,
                source=read.path,
            )
        path = Path(args.path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        if not path.exists() or not path.is_file():
            raise SystemExit(f"MCP server manifest does not exist: {path}")
        return runtime.mcp.register_server_from_yaml_text(
            path.read_text(encoding="utf-8"),
            actor=actor,
            replace=args.replace,
            require_capability=False,
            source=str(path),
        )
    if command == "list":
        return runtime.mcp.list_servers(
            actor=actor if require_capability else None,
            require_capability=require_capability,
            text=args.text,
            limit=args.limit,
        )
    if command == "inspect":
        return runtime.mcp.inspect_server(
            args.server_id,
            actor=actor if require_capability else None,
            require_capability=require_capability,
            include_sensitive_fields=not require_capability,
        )
    if command == "discover":
        return to_jsonable(
            runtime.mcp.discover(
                args.server_id,
                actor=actor,
                require_capability=require_capability,
            )
        )
    if command == "tools":
        return runtime.mcp.list_tools(
            args.server_id,
            actor=actor,
            require_capability=require_capability,
            refresh=args.refresh,
        )
    if command == "call":
        if args.actor_pid is not None and args.actor_pid != args.pid:
            raise SystemExit("mcp call --actor-pid must match the target process pid")
        arguments = _parse_json_value(args.arguments_json) if args.arguments_json is not None else {}
        return to_jsonable(runtime.mcp.call_tool(args.pid, args.server_id, args.tool_id, arguments))
    if command == "unregister":
        return runtime.mcp.unregister_server(
            args.server_id,
            actor=actor,
            require_capability=require_capability,
        )
    raise SystemExit(f"unknown mcp command: {command}")


def _add_modules_parser_args(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="modules_command", required=True)
    list_parser = sub.add_parser("list", help="List loaded startup runtime modules")
    list_parser.add_argument("--limit", type=int)
    inspect = sub.add_parser("inspect", help="Inspect one loaded startup runtime module")
    inspect.add_argument("module_id")
    verify = sub.add_parser("verify", help="Verify a module manifest without loading it")
    verify.add_argument("path")


def _run_modules_command(runtime: Runtime, args: argparse.Namespace) -> dict[str, Any] | list[dict[str, Any]]:
    command = args.modules_command
    if command == "list":
        return runtime.modules.list_modules(limit=args.limit)
    if command == "inspect":
        return runtime.modules.inspect_module(args.module_id)
    if command == "verify":
        return runtime.modules.verify_manifest(
            args.path,
            trusted_modules=args.trusted_module,
            trusted_sha256=args.trusted_module_sha256,
        )
    raise SystemExit(f"unknown modules command: {command}")


def _capability_spec_from_args(args: argparse.Namespace) -> CapabilitySpec:
    return CapabilitySpec(
        resource=args.resource,
        rights={str(right) for right in args.rights},
        effect=CapabilityEffect(args.effect),
        constraints=_parse_json_mapping(args.constraints_json, "--constraints-json"),
        metadata=_parse_json_mapping(args.metadata_json, "--metadata-json"),
        expires_at=args.expires_at,
        uses_remaining=args.uses_remaining,
        delegable=args.delegable,
        revocable=args.revocable,
    )


def _message_cli_summary(message: ProcessMessage) -> dict[str, Any]:
    return {
        "message_id": message.message_id,
        "sender": message.sender,
        "recipient_pid": message.recipient_pid,
        "kind": message.kind.value,
        "channel": message.channel,
        "correlation_id": message.correlation_id,
        "reply_to": message.reply_to,
        "subject": message.subject,
        "body": message.body,
        "payload": message.payload,
        "status": message.status.value,
        "created_at": message.created_at,
    }


def _single_active_process_pid(runtime: Runtime) -> str | None:
    active = [process.pid for process in runtime.process.list() if process.status not in _TERMINAL_PROCESS_STATUSES]
    return active[0] if len(active) == 1 else None


def _redirect_human_output_to_stderr(runtime: Runtime) -> None:
    provider = runtime.substrate.human
    if hasattr(provider, "output_sink"):
        provider.output_sink = lambda message: print(message, file=sys.stderr, flush=True)


def _start_interactive_input_thread(
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[str | None],
    stop: threading.Event,
) -> None:
    def worker() -> None:
        while not stop.is_set():
            print("agent-libos> ", end="", file=sys.stderr, flush=True)
            line = sys.stdin.readline()
            if line == "":
                _enqueue_interactive_line(loop, queue, None)
                return
            _enqueue_interactive_line(loop, queue, line.rstrip("\r\n"))

    threading.Thread(target=worker, name="agent-libos-cli-input", daemon=True).start()


def _enqueue_interactive_line(
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[str | None],
    line: str | None,
) -> None:
    try:
        loop.call_soon_threadsafe(queue.put_nowait, line)
    except RuntimeError:
        pass


def _print_interactive_help(target_pid: str) -> None:
    print(
        (
            f"Interactive human input target: {target_pid}\n"
            "Plain text sends a normal message, or answers the currently displayed Human request.\n"
            "Process commands: /message <text>, /interrupt <text>, /pid <pid>, /help, /exit.\n"
            "Displayed-request commands: /answer <text>, /approve, /reject, /allow, /ask."
        ),
        file=sys.stderr,
        flush=True,
    )


def _drain_interactive_queue(
    runtime: Runtime,
    queue: asyncio.Queue[str | None],
    state: dict[str, str],
    human: str,
    channel: str,
    posted: list[dict[str, Any]],
) -> str | None:
    command: str | None = None
    while True:
        try:
            line = queue.get_nowait()
        except asyncio.QueueEmpty:
            return command
        command = _handle_interactive_line(runtime, line, state, human, channel, posted)
        if command in {"exit", "eof"}:
            return command


def _handle_interactive_line(
    runtime: Runtime,
    line: str | None,
    state: dict[str, str],
    human: str,
    channel: str,
    posted: list[dict[str, Any]],
) -> str | None:
    if line is None:
        return "eof"
    if _handle_interactive_human_response(
        runtime,
        line,
        human,
        shown_request_id=state.get("shown_request_id"),
    ):
        return None
    parsed = _parse_interactive_line(line)
    command = parsed.get("command")
    if command is None:
        return None
    if command == "exit":
        return "exit"
    if command == "help":
        _print_interactive_help(state["pid"])
        return None
    if command == "pid":
        pid = str(parsed["pid"])
        runtime.process.get(pid)
        state["pid"] = pid
        print(f"Target process: {pid}", file=sys.stderr, flush=True)
        return None
    if command == "message":
        kind = ProcessMessageKind(str(parsed["kind"]))
        message = runtime.human.send_process_message(
            state["pid"],
            str(parsed["body"]),
            kind=kind,
            human=human,
            channel=channel,
            payload={"source": "cli.interactive"},
        )
        summary = _message_cli_summary(message)
        posted.append(summary)
        print(f"Sent {message.kind.value} message {message.message_id} -> {message.recipient_pid}", file=sys.stderr, flush=True)
        return None
    return None


def _parse_interactive_line(line: str) -> dict[str, Any]:
    stripped = line.strip()
    if not stripped:
        return {}
    if not stripped.startswith("/"):
        return {"command": "message", "kind": ProcessMessageKind.NORMAL.value, "body": stripped}
    command, _, rest = stripped[1:].partition(" ")
    command = command.lower()
    body = rest.strip()
    if command in {"exit", "quit", "q"}:
        return {"command": "exit"}
    if command in {"help", "h", "?"}:
        return {"command": "help"}
    if command in {"pid", "target"}:
        if not body:
            raise SystemExit("/pid requires a process id")
        return {"command": "pid", "pid": body}
    if command in {"interrupt", "i"}:
        return {
            "command": "message",
            "kind": ProcessMessageKind.INTERRUPT.value,
            "body": body or "Human requested attention.",
        }
    if command in {"message", "m"}:
        return {"command": "message", "kind": ProcessMessageKind.NORMAL.value, "body": body}
    print(f"Unknown interactive command: /{command}. Type /help for commands.", file=sys.stderr, flush=True)
    return {}


def _process_interactive_terminal_outputs(runtime: Runtime, human: str) -> list[Any]:
    processed: list[Any] = []
    while True:
        pending = runtime.human.pending(human=human)
        if not pending or pending[0].payload.get("type") != "output":
            return processed
        processed.append(runtime.human.process_next_terminal(human=human))


def _show_pending_interactive_human_request(runtime: Runtime, human: str, state: dict[str, str]) -> None:
    retained_release_id = state.get("pending_release_request_id")
    request = _interactive_input_request_by_id(
        runtime,
        human,
        retained_release_id,
    )
    if request is None:
        state["pending_release_request_id"] = ""
        request = _first_interactive_input_request(runtime, human)
    if request is None:
        state["shown_request_id"] = ""
        return
    if state.get("shown_request_id") == request.request_id:
        return
    request_type = str(request.payload.get("type") or "approval")
    if request_type == "permission_request":
        suffix = "Reply a=always allow, d=always deny, e=ask each time."
    elif request_type == "question":
        suffix = "Reply with the answer text."
    else:
        suffix = "Reply y/yes to approve, n/no to reject."
    try:
        runtime.human.present_terminal_request(request, suffix=suffix)
    except HumanApprovalRequired as exc:
        # The protected Human Sink created a metadata-only release request.
        # Retain the exact prerequisite because bounded pending-request windows
        # are allowed to omit a newly-created release behind older requests.
        state["pending_release_request_id"] = exc.request_id
        state["shown_request_id"] = ""
        return
    state["shown_request_id"] = request.request_id


def _handle_interactive_human_response(
    runtime: Runtime,
    line: str,
    human: str,
    *,
    shown_request_id: str | None,
) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith(("/message", "/m", "/interrupt", "/i", "/pid", "/target", "/help", "/exit", "/quit")):
        return False
    request = _interactive_input_request_by_id(
        runtime,
        human,
        shown_request_id,
    )
    if request is None:
        return False
    response = _interactive_response_text(stripped)
    if response is None:
        return False
    request_type = request.payload.get("type")
    if request_type == "question":
        runtime.human.approve(
            request.request_id,
            {"approved": True, "answer": response, "source": "interactive_cli"},
            responder=f"human:{human}",
        )
        print(f"Answered human request {request.request_id}", file=sys.stderr, flush=True)
        return True
    if request_type == "permission_request":
        policy = _interactive_permission_policy(response)
        if policy is None:
            print(
                "Permission response must be /allow, /ask, or /reject; request remains pending.",
                file=sys.stderr,
                flush=True,
            )
            return True
        decision = {"policy": policy, "source": "interactive_cli"}
        if policy == CapabilityManager.ALWAYS_DENY:
            runtime.human.reject(request.request_id, {"approved": False, **decision}, responder=f"human:{human}")
        else:
            runtime.human.approve(request.request_id, {"approved": True, **decision}, responder=f"human:{human}")
        print(f"Resolved permission request {request.request_id} with policy={policy}", file=sys.stderr, flush=True)
        return True
    normalized = response.strip().lower()
    if normalized in {"y", "yes", "approve", "approved"}:
        approved = True
    elif normalized in {"n", "no", "reject", "rejected", "deny", "denied"}:
        approved = False
    else:
        print(
            "Approval response must be /approve or /reject; request remains pending.",
            file=sys.stderr,
            flush=True,
        )
        return True
    if approved:
        runtime.human.approve(
            request.request_id,
            {"approved": True, "source": "interactive_cli"},
            responder=f"human:{human}",
        )
    else:
        runtime.human.reject(
            request.request_id,
            {"approved": False, "source": "interactive_cli"},
            responder=f"human:{human}",
        )
    print(f"{'Approved' if approved else 'Rejected'} human request {request.request_id}", file=sys.stderr, flush=True)
    return True


def _interactive_response_text(stripped: str) -> str | None:
    if not stripped.startswith("/"):
        return stripped
    command, _, rest = stripped[1:].partition(" ")
    command = command.lower()
    if command in {"answer", "reply"}:
        return rest
    if command in {"approve", "yes", "y"}:
        return "yes"
    if command in {"reject", "deny", "no", "n"}:
        return "no"
    if command in {"allow", "always-allow", "always_allow"}:
        return "always_allow"
    if command in {"ask", "ask-each-time", "ask_each_time"}:
        return "ask_each_time"
    return None


def _interactive_permission_policy(answer: str) -> str | None:
    normalized = answer.strip().lower()
    return {
        "a": CapabilityManager.ALWAYS_ALLOW,
        "allow": CapabilityManager.ALWAYS_ALLOW,
        "always_allow": CapabilityManager.ALWAYS_ALLOW,
        "yes": CapabilityManager.ALWAYS_ALLOW,
        "y": CapabilityManager.ALWAYS_ALLOW,
        "e": CapabilityManager.ASK_EACH_TIME,
        "ask": CapabilityManager.ASK_EACH_TIME,
        "each": CapabilityManager.ASK_EACH_TIME,
        "ask_each_time": CapabilityManager.ASK_EACH_TIME,
        "d": CapabilityManager.ALWAYS_DENY,
        "deny": CapabilityManager.ALWAYS_DENY,
        "always_deny": CapabilityManager.ALWAYS_DENY,
        "no": CapabilityManager.ALWAYS_DENY,
        "n": CapabilityManager.ALWAYS_DENY,
    }.get(normalized)


def _first_interactive_input_request(runtime: Runtime, human: str) -> Any | None:
    pending = [
        request
        for request in runtime.human.pending(human=human)
        if request.payload.get("type") != "output"
    ]
    return next(
        (
            request
            for request in pending
            if request.payload.get("type") == "data_release_approval"
        ),
        pending[0] if pending else None,
    )


def _interactive_input_request_by_id(
    runtime: Runtime,
    human: str,
    request_id: str | None,
) -> Any | None:
    """Resolve only the still-pending request that the CLI retained or showed."""

    if not request_id:
        return None
    try:
        request = runtime.human.get(request_id)
    except NotFound:
        return None
    if (
        request.human != human
        or request.status is not HumanRequestStatus.PENDING
        or request.payload.get("type") == "output"
    ):
        return None
    return request


def _process_cli_summary(process: Any) -> dict[str, Any]:
    return {
        "pid": process.pid,
        "image": process.image_id,
        **process_state_to_mapping(
            process.status.value,
            process.wait_state,
            process.outcome,
            process.state_generation,
        ),
        "goal_oid": process.goal_oid,
        "working_directory": process.working_directory,
        "active_tools": sorted(process.tool_table),
    }


def run_demo(runtime: Runtime) -> dict[str, Any]:
    runtime_defaults = runtime.config.runtime
    tool_sequence: list[dict[str, Any]] = []
    root = runtime.process.spawn(
        image=runtime_defaults.coding_image_id,
        goal={"text": "Fix failing tests in this repository"},
    )
    log = """
    =========================== FAILURES ===========================
    FAILED tests/test_math.py::test_add - AssertionError: assert 5 == 4
    E   AssertionError: assert 5 == 4
    ==================== 1 failed, 3 passed in 0.12s ====================
    """.strip()
    log_handle = runtime.memory.create_object(
        pid=root,
        object_type=ObjectType.ERROR_TRACE,
        payload={"log": log},
        metadata=ObjectMetadata(title="pytest failure log", tags=["pytest", "failure"]),
    )
    root_proc = runtime.process.get(root)
    assert root_proc.memory_view is not None
    runtime.store.append_process_memory_roots(root, [log_handle])

    worker = runtime.process.fork(
        parent=root,
        goal={"text": "Analyze the pytest failure log"},
        memory_view=MemoryViewSpec(roots=[log_handle], mode=ViewMode.READ_ONLY),
        mode=ForkMode.WORKER,
    )
    parse_tool = runtime.tools.resolve("parse_pytest_log")
    parsed = runtime.tools.call(worker, parse_tool, {"log": log})
    tool_sequence.append(_tool_call_summary("parse_pytest_log", worker, parsed))
    runtime.process.exit(worker, parsed.result_handle)
    worker_result = runtime.process.wait(root, worker)
    if worker_result.result is not None:
        root_proc = runtime.process.get(root)
        if root_proc.memory_view is not None:
            runtime.store.append_process_memory_roots(root, [worker_result.result])

    jit_source = """
export function run(args, libos) {
  const log = String(args.log ?? "");
  const names = [];
  for (const rawLine of log.split("\\n")) {
    const line = rawLine.trim();
    if (line.startsWith("FAILED ")) {
      names.push(line.split(/\\s+/)[1]);
    }
  }
  return { tests: names, count: names.length };
}
""".strip()
    candidate = runtime.tools.propose(
        root,
        {
            "name": "extract_failed_tests",
            "description": "Extract failed pytest node ids.",
            "input_schema": {"type": "object", "properties": {"log": {"type": "string"}}},
            "output_schema": {"type": "object"},
        },
        source_code=jit_source,
        tests=[{"args": {"log": log}, "expected": {"tests": ["tests/test_math.py::test_add"], "count": 1}}],
    )
    validation = runtime.tools.validate(candidate)
    jit_tool = runtime.tools.register(root, candidate) if validation.ok else None
    jit_result = runtime.tools.call(root, jit_tool, {"log": log}) if jit_tool is not None else None
    if jit_result is not None:
        tool_sequence.append(_tool_call_summary("extract_failed_tests", root, jit_result))

    checkpoint = runtime.checkpoint.checkpoint(root, "before high-risk patch application")
    write_args = {
        "path": DEMO_PATCH_PREVIEW_PATH,
        "content": DEMO_PATCH_PREVIEW_CONTENT,
        "overwrite": True,
    }
    denied_without_filesystem = runtime.tools.call(root, "write_text_file", write_args)
    tool_sequence.append(_tool_call_summary("write_text_file", root, denied_without_filesystem))
    if denied_without_filesystem.ok:
        raise RuntimeError("demo expected write_text_file to fail before filesystem write capability was granted")

    filesystem_resource = runtime.filesystem.resource_for(DEMO_PATCH_PREVIEW_PATH)
    approval_request = runtime.human.query(
        pid=root,
        human=runtime_defaults.default_human,
        request={
            "type": "approval",
            "question": f"Grant workspace write capability for {DEMO_PATCH_PREVIEW_PATH}?",
            "requested_capability": {
                "subject": root,
                "resource": filesystem_resource,
                "rights": [CapabilityRight.WRITE.value],
            },
            "context": {"path": DEMO_PATCH_PREVIEW_PATH, "tool": "write_text_file"},
        },
        blocking=True,
    )
    runtime.human.approve(approval_request, {"approved": True, "reason": "demo filesystem write approval"})
    approved_call = runtime.tools.call(root, "write_text_file", write_args)
    tool_sequence.append(_tool_call_summary("write_text_file", root, approved_call))
    target = runtime.workspace_root / DEMO_PATCH_PREVIEW_PATH
    target_exists = target.exists()
    target_content_matches = target_exists and target.read_text(encoding="utf-8") == DEMO_PATCH_PREVIEW_CONTENT
    if not approved_call.ok or not target_content_matches:
        raise RuntimeError("demo write_text_file contract failed")

    audit_records = runtime.audit.trace()
    audit_summary = dict(Counter(record.action for record in audit_records))
    report_payload = {
        "summary": (
            "Analyzed a failing pytest log, extracted the failed test, checkpointed before the write, "
            "verified filesystem denial before external-resource authorization, requested human approval for workspace write, "
            "and wrote a patch preview file."
        ),
        "problem": {
            "failed_test": "tests/test_math.py::test_add",
            "assertion": "AssertionError: assert 5 == 4",
            "source": "synthetic pytest failure log",
        },
        "evidence": [
            {"kind": "pytest_log", "oid": log_handle.oid, "title": "pytest failure log"},
            {
                "kind": "worker_parse_result",
                "oid": worker_result.result.oid if worker_result.result else None,
                "payload": parsed.payload,
            },
            {
                "kind": "jit_extract_result",
                "payload": jit_result.payload if jit_result else None,
                "validation_ok": validation.ok,
                "validation_errors": validation.errors,
            },
            {"kind": "filesystem_denial", "payload": denied_without_filesystem.payload, "error": denied_without_filesystem.error},
        ],
        "tool_sequence": tool_sequence,
        "authorization": {
            "filesystem_write_approval_request": approval_request,
            "filesystem_write_resource": filesystem_resource,
            "filesystem_write_granted_by": runtime_defaults.default_human_actor,
            "filesystem_write_denied_before_grant": {
                "ok": denied_without_filesystem.ok,
                "error": denied_without_filesystem.error,
            },
        },
        "external_side_effects": [
            {
                "adapter": "filesystem",
                "action": "write_text",
                "path": DEMO_PATCH_PREVIEW_PATH,
                "bytes_written": approved_call.payload.get("bytes_written") if isinstance(approved_call.payload, dict) else None,
                "audit_action": "primitive.filesystem.write_text",
            }
        ],
        "checkpoint": checkpoint,
        "write_result": _tool_call_summary("write_text_file", root, approved_call),
        "target_file": {
            "path": DEMO_PATCH_PREVIEW_PATH,
            "exists": target_exists,
            "content_matches": target_content_matches,
        },
        "audit_summary": audit_summary,
        "limits": "This demo writes a patch preview only; it is not a production automatic repair system.",
        "next_steps": [
            "Review the patch preview.",
            "Add real repository tests for the suspected math assertion.",
            "Implement policy decisions for side-effect tools before expanding external adapters.",
        ],
    }
    report_handle = runtime.memory.create_object(
        pid=root,
        object_type=ObjectType.SUMMARY,
        payload=report_payload,
        metadata=ObjectMetadata(title="coding-agent demo final report", tags=["demo", "report"]),
    )
    runtime.process.exit(root, report_handle)
    final_audit_records = runtime.audit.trace()
    final_audit_summary = dict(Counter(record.action for record in final_audit_records))
    return {
        "root": root,
        "worker": worker,
        "worker_result_oid": worker_result.result.oid if worker_result.result else None,
        "jit_candidate": candidate,
        "jit_validation_ok": validation.ok,
        "jit_validation_errors": validation.errors,
        "jit_result": jit_result.payload if jit_result else None,
        "approval_request": approval_request,
        "filesystem_write_denial": _tool_call_summary("write_text_file", root, denied_without_filesystem),
        "checkpoint": checkpoint,
        "tool_sequence": tool_sequence,
        "write_result": _tool_call_summary("write_text_file", root, approved_call),
        "write_path": DEMO_PATCH_PREVIEW_PATH,
        "target_file_exists": target_exists,
        "target_file_content_matches": target_content_matches,
        "final_report_oid": report_handle.oid,
        "final_report": report_payload,
        "audit_records": len(final_audit_records),
        "audit_summary": final_audit_summary,
    }


def _tool_call_summary(tool: str, pid: str, result: ToolCallResult) -> dict[str, Any]:
    return {
        "pid": pid,
        "tool": tool,
        "ok": result.ok,
        "tool_id": result.tool_id,
        "call_id": result.call_id,
        "result_oid": result.result_handle.oid if result.result_handle else None,
        "payload": result.payload,
        "error": result.error,
    }


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    cli()
