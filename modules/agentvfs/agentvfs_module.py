"""agentvfs trusted Runtime Module.

Exposes model-facing tools (agentvfs_status, agentvfs_checkpoint,
agentvfs_rollback) over one Host-bound agentvfs workspace control socket.
The workspace binding is Host-only composition: the Host sets the substrate
attribute ``agentvfs`` (workspace name, or mapping with ``workspace`` and
optional explicit ``socket``); the startup hook attaches to the already-running
workspace via session.json discovery and fails closed when it is not running.
The module never starts or stops the FUSE daemon. Tools enforce capability
authority on ``agentvfs:<workspace>`` before any socket traffic: READ for
status, WRITE for checkpoint, ADMIN for the destructive rollback.

Paired mode gives one tool call two-plane semantics. A paired checkpoint
snapshots the agentvfs filesystem first, then records a libOS checkpoint whose
metadata embeds the agentvfs commit hash (order forced by the metadata), after
probing the process's self-checkpoint authority so a denial leaves no orphan
on either plane. A paired rollback rolls the filesystem back first, validates
the named libOS checkpoint pairs with the resulting commit, then attempts the
legitimate in-tool restore (``CheckpointManager.restore`` refuses while the
scheduler runs a quantum, so a model-invoked call reports
``libos_restore=pending_host_restore`` with a Host hint instead of bypassing
quiescence; a Host-driven call with admin authority restores both planes).
"""

from __future__ import annotations

import json
import os
import re
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent_libos.models import AgentImage, CapabilityRight, EventType
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.tools.base import (
    SyncAgentTool,
    ToolContext,
    ToolErrorCode,
    ToolExecutionError,
    ToolPolicy,
)

_ADAPTER_ATTR = "_agent_libos_agentvfs_adapter"
_UNBOUND = "unbound"
_WORKSPACE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
MODULE_ID = "agent-libos-agentvfs:v0"

LIBOS_RESTORE_RESTORED = "restored"
LIBOS_RESTORE_PENDING = "pending_host_restore"
LIBOS_RESTORE_SKIPPED = "skipped"
_RESTORE_REFUSED_PREFIX = "refused while scheduler"
_HOST_RESTORE_HINT = (
    "Host must run runtime.checkpoint.restore(actor, checkpoint_id, "
    "require_capability=False) for the paired checkpoint once the process is quiescent"
)


class AgentVfsControlError(RuntimeError):
    """The agentvfs control daemon rejected or failed one request."""

    def __init__(self, request: str, detail: str) -> None:
        super().__init__(f"agentvfs control request {request!r} failed: {detail}")
        self.request = request
        self.detail = detail


class AgentVfsControlClient:
    """Newline-delimited JSON AF_UNIX client for one agentvfs daemon socket."""

    def __init__(self, socket_path: str | Path, *, timeout_s: float = 30.0) -> None:
        self.socket_path = str(socket_path)
        self.timeout_s = timeout_s

    def request(self, line: str) -> dict[str, Any]:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self.timeout_s)
            sock.connect(self.socket_path)
            sock.sendall((line + "\n").encode())
            buffer = bytearray()
            while not buffer.endswith(b"\n"):
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buffer += chunk
        text = bytes(buffer).decode(errors="replace").strip()
        if not text:
            raise AgentVfsControlError(line, "empty response")
        try:
            response = json.loads(text)
        except ValueError as exc:
            raise AgentVfsControlError(line, f"invalid JSON reply: {text!r}") from exc
        if not isinstance(response, dict) or not response.get("ok"):
            detail = response.get("error") if isinstance(response, dict) else text
            raise AgentVfsControlError(line, str(detail))
        return response


@dataclass(frozen=True)
class AgentVfsBinding:
    """Host-owned attachment to one running agentvfs workspace."""

    workspace: str
    socket: str | None = None
    request_timeout_s: float = 30.0


def _coerce_binding(value: Any) -> AgentVfsBinding:
    if isinstance(value, AgentVfsBinding):
        binding = value
    elif isinstance(value, str):
        binding = AgentVfsBinding(workspace=value)
    elif isinstance(value, dict):
        binding = AgentVfsBinding(**value)
    else:
        binding = AgentVfsBinding(
            **{
                field_name: getattr(value, field_name)
                for field_name in AgentVfsBinding.__dataclass_fields__
                if hasattr(value, field_name)
            }
        )
    if not _WORKSPACE_NAME.fullmatch(binding.workspace):
        raise ValidationError(
            f"agentvfs workspace name {binding.workspace!r} must match [A-Za-z0-9._-]{{1,80}}"
        )
    if binding.request_timeout_s <= 0:
        raise ValidationError("agentvfs request_timeout_s must be positive")
    return binding


def _runtime_root() -> Path:
    """Mirror agentvfs default_workspace_root(): $XDG_RUNTIME_DIR/agentvfs
    when set, otherwise /tmp/agentvfs-<uid>."""
    base = os.environ.get("XDG_RUNTIME_DIR")
    return Path(base) / "agentvfs" if base else Path(f"/tmp/agentvfs-{os.getuid()}")


def _discover_socket(binding: AgentVfsBinding) -> str:
    if binding.socket:
        return str(binding.socket)
    session_path = _runtime_root() / binding.workspace / "session.json"
    try:
        record = json.loads(session_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValidationError(
            f"agentvfs workspace {binding.workspace!r} is not attachable: "
            f"cannot read session file {session_path}"
        ) from exc
    socket_path = record.get("socket")
    if record.get("status") != "started" or not socket_path:
        raise ValidationError(
            f"agentvfs workspace {binding.workspace!r} is not running "
            f"(session status={record.get('status')!r})"
        )
    return str(socket_path)


class AgentVfsAdapter:
    """Capability-gated bridge from process authority to the control socket."""

    def __init__(self, host: Any, binding: AgentVfsBinding) -> None:
        self.host = host
        self.binding = binding
        self.resource = f"agentvfs:{binding.workspace}"
        self.client = AgentVfsControlClient(
            _discover_socket(binding), timeout_s=binding.request_timeout_s
        )

    def _require_right(self, pid: str, right: CapabilityRight, operation: str) -> None:
        decision = self.host.capability.authorize(pid, self.resource, right)
        if not decision.allowed:
            self.host.audit.record(
                actor=pid,
                action=f"module.agentvfs.{operation}.denied",
                target=self.resource,
                decision={"right": right.value, "reason": decision.reason},
            )
            raise CapabilityDenied(
                f"{pid} denied agentvfs {operation} on {self.resource}: {decision.reason}"
            )

    def _record(self, pid: str, operation: str, event_type: EventType) -> None:
        self.host.audit.record(
            actor=pid,
            action=f"module.agentvfs.{operation}",
            target=self.resource,
            decision={"workspace": self.binding.workspace, "operation": operation},
        )
        self.host.events.emit(
            event_type,
            source=pid,
            target=self.resource,
            payload={"workspace": self.binding.workspace, "operation": operation},
        )

    def status(self, pid: str) -> dict[str, Any]:
        self._require_right(pid, CapabilityRight.READ, "status")
        response = self.client.request("status")
        self._record(pid, "status", EventType.EXTERNAL_READ)
        return response

    def checkpoint(self, pid: str, label: str) -> dict[str, Any]:
        self._require_right(pid, CapabilityRight.WRITE, "checkpoint")
        response = self.client.request(f"checkpoint {label}")
        self._record(pid, "checkpoint", EventType.EXTERNAL_WRITE)
        return response

    def rollback(self, pid: str, target: str) -> dict[str, Any]:
        self._require_right(pid, CapabilityRight.ADMIN, "rollback")
        response = self.client.request(f"rollback {target}")
        self._record(pid, "rollback", EventType.EXTERNAL_WRITE)
        return response

    def detach(self) -> bool:
        """Shutdown finalizer: release module state without stopping the daemon."""
        self.host.audit.record(
            actor=f"module:{MODULE_ID}",
            action="module.agentvfs.detach",
            target=self.resource,
            decision={"workspace": self.binding.workspace},
        )
        return True


def initialize_agentvfs(runtime: Any) -> None:
    if runtime.get_runtime_attribute(_ADAPTER_ATTR) is not None:
        return
    binding_value = getattr(runtime.substrate, "agentvfs", None)
    if binding_value is None:
        # Inert without a Host binding: the tools stay registered but fail
        # closed at call time because no socket path exists to talk to.
        runtime.set_runtime_attribute(_ADAPTER_ATTR, _UNBOUND)
        return
    binding = _coerce_binding(binding_value)
    adapter = AgentVfsAdapter(runtime, binding)
    runtime.set_runtime_attribute(_ADAPTER_ATTR, adapter)
    runtime.bind_shutdown_finalizer(adapter.detach)


def _adapter(ctx: ToolContext) -> AgentVfsAdapter:
    runtime = _runtime(ctx)
    adapter = runtime.module_state.get(_ADAPTER_ATTR)
    if adapter == _UNBOUND:
        raise ToolExecutionError(
            "No agentvfs workspace is bound; the Host must set the substrate "
            "'agentvfs' binding before this tool can run.",
            code=ToolErrorCode.EXECUTION_ERROR,
            retryable=False,
        )
    if adapter is None:
        raise ToolExecutionError(
            "agentvfs module has not initialized.",
            code=ToolErrorCode.EXECUTION_ERROR,
            retryable=False,
        )
    return adapter


def _runtime(ctx: ToolContext) -> Any:
    if ctx.runtime is None:
        raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
    return ctx.runtime


def _validate_label(value: str, field: str) -> str:
    if not _WORKSPACE_NAME.fullmatch(value):
        raise ValidationError(
            f"agentvfs {field} {value!r} must match [A-Za-z0-9._-]{{1,80}}"
        )
    return value


def _require_libos_checkpoint_right(runtime: Any, pid: str) -> None:
    """Probe the process's self-checkpoint authority before agentvfs traffic."""
    resource = f"checkpoint:process:{pid}"
    decision = runtime.capability.authorize(pid, resource, CapabilityRight.WRITE)
    if not decision.allowed:
        runtime.audit.record(
            actor=pid,
            action="module.agentvfs.pair_checkpoint.denied",
            target=resource,
            decision={"right": CapabilityRight.WRITE.value, "reason": decision.reason},
        )
        raise CapabilityDenied(
            f"{pid} denied paired libOS checkpoint on {resource}: {decision.reason}"
        )


def _create_paired_libos_checkpoint(
    runtime: Any,
    pid: str,
    workspace: str,
    label: str,
    commit: str,
    reason: str | None,
) -> str:
    try:
        return runtime.checkpoint.create(
            pid,
            reason or f"agentvfs checkpoint {label}",
            actor=pid,
            metadata={
                "agentvfs_workspace": workspace,
                "agentvfs_label": label,
                "agentvfs_commit": commit,
            },
        )
    except Exception as exc:
        raise ToolExecutionError(
            f"agentvfs checkpoint {commit} succeeded but the paired libOS "
            f"checkpoint failed: {exc}",
            code=ToolErrorCode.EXECUTION_ERROR,
            retryable=False,
            details={
                "agentvfs_commit": commit,
                "agentvfs_label": label,
                "pairing": "libos_checkpoint_failed",
            },
        ) from exc


def _inspect_paired_checkpoint(
    runtime: Any, pid: str, checkpoint_id: str
) -> dict[str, Any]:
    """Load the paired checkpoint's metadata before any socket traffic."""
    inspected = runtime.checkpoint.inspect(checkpoint_id, actor=pid)
    summary = inspected["checkpoint"]
    if summary["pid"] != pid:
        raise ValidationError(
            f"paired libOS checkpoint {checkpoint_id} belongs to "
            f"{summary['pid']}, not {pid}"
        )
    return summary.get("metadata") or {}


def _require_commit_pairing(
    metadata: dict[str, Any], checkpoint_id: str, rolled_to: str
) -> None:
    paired_commit = metadata.get("agentvfs_commit")
    if paired_commit != rolled_to:
        raise ValidationError(
            f"paired libOS checkpoint {checkpoint_id} references agentvfs "
            f"commit {paired_commit} but the workspace rolled back to {rolled_to}; "
            "the filesystem rollback stands, re-pair with the matching checkpoint"
        )


def _restore_libos_paired(
    runtime: Any, pid: str, workspace: str, checkpoint_id: str
) -> tuple[str, str | None]:
    """Attempt the process-authorized libOS restore, or report a pending Host
    restore when authority or scheduler quiescence is missing."""
    decision = runtime.capability.authorize(
        pid, f"checkpoint:{checkpoint_id}", CapabilityRight.ADMIN
    )
    if not decision.allowed:
        hint = (
            f"{_HOST_RESTORE_HINT}; process lacks admin authority on "
            f"checkpoint:{checkpoint_id}"
        )
        return _pending_restore(
            runtime, pid, workspace, checkpoint_id, "authority", hint
        )
    try:
        runtime.checkpoint.restore(pid, checkpoint_id)
    except ValidationError as exc:
        if _RESTORE_REFUSED_PREFIX in str(exc):
            hint = f"{_HOST_RESTORE_HINT}; scheduler refused the in-tool restore: {exc}"
            return _pending_restore(
                runtime, pid, workspace, checkpoint_id, "scheduler_busy", hint
            )
        raise
    return LIBOS_RESTORE_RESTORED, None


def _pending_restore(
    runtime: Any,
    pid: str,
    workspace: str,
    checkpoint_id: str,
    reason: str,
    hint: str,
) -> tuple[str, str | None]:
    runtime.audit.record(
        actor=pid,
        action="module.agentvfs.libos_restore_pending",
        target=f"checkpoint:{checkpoint_id}",
        decision={"reason": reason, "workspace": workspace},
    )
    return LIBOS_RESTORE_PENDING, hint


class AgentvfsStatusArgs(BaseModel):
    pass


class AgentvfsStatusOutput(BaseModel):
    workspace: str
    commit: str | None = None
    branch: str | None = None
    version: str | None = None
    telemetry_drops_total: int | None = None


class AgentvfsCheckpointArgs(BaseModel):
    label: str = Field(description="Checkpoint label, charset [A-Za-z0-9._-], 1-80 chars.")
    pair_libos: bool = Field(
        default=False,
        description=(
            "Also create a libOS checkpoint of this process whose metadata records "
            "the agentvfs commit hash, so a later paired rollback can restore both "
            "the filesystem and libOS object/SQL state."
        ),
    )
    reason: str | None = Field(
        default=None,
        description="Optional reason recorded on the paired libOS checkpoint.",
    )


class AgentvfsCheckpointOutput(BaseModel):
    commit: str
    label: str
    libos_checkpoint_id: str | None = Field(
        default=None,
        description="Present only for paired checkpoints: the libOS checkpoint holding this commit hash in its metadata.",
    )


class AgentvfsRollbackArgs(BaseModel):
    target: str = Field(
        description="Rollback target: a checkpoint label or 64-hex commit hash."
    )
    pair_libos: bool = Field(
        default=False,
        description=(
            "Validate the named paired libOS checkpoint (its metadata must embed the "
            "commit actually rolled back to) and attempt its restore after the "
            "filesystem rollback. The libOS restore is Host-mediated when the process "
            "lacks checkpoint admin authority or the scheduler is running; the result "
            "then reports libos_restore=pending_host_restore with a hint instead of "
            "restoring in-tool."
        ),
    )
    libos_checkpoint_id: str | None = Field(
        default=None,
        description="Required when pair_libos is true: the checkpoint id returned by a paired agentvfs_checkpoint.",
    )


class AgentvfsRollbackOutput(BaseModel):
    rolled_back_to: str
    paired_libos_checkpoint_id: str | None = None
    libos_restore: str = Field(
        description="restored | pending_host_restore | skipped (non-paired rollback)."
    )
    libos_restore_hint: str | None = Field(
        default=None,
        description="Present when libos_restore=pending_host_restore: what the Host must run.",
    )


class AgentvfsStatusTool(SyncAgentTool[AgentvfsStatusArgs]):
    name = "agentvfs_status"
    description = (
        "Report the Host-bound agentvfs workspace daemon status (branch, commit, "
        "version). Requires agentvfs read capability; never mutates state."
    )
    args_schema = AgentvfsStatusArgs
    output_schema = AgentvfsStatusOutput
    policy = ToolPolicy(
        side_effects=False,
        idempotent=True,
        declared_permissions={"agentvfs.read"},
        timeout_s=None,
    )
    tags = ["agentvfs", "checkpoint", "inspect"]

    def run(self, args: AgentvfsStatusArgs, ctx: ToolContext) -> AgentvfsStatusOutput:
        adapter = _adapter(ctx)
        response = adapter.status(ctx.pid)
        return AgentvfsStatusOutput(
            workspace=adapter.binding.workspace,
            commit=response.get("commit"),
            branch=response.get("branch"),
            version=response.get("version"),
            telemetry_drops_total=response.get("telemetry_drops_total"),
        )


class AgentvfsCheckpointTool(SyncAgentTool[AgentvfsCheckpointArgs]):
    name = "agentvfs_checkpoint"
    description = (
        "Snapshot the process's agentvfs workspace filesystem into the "
        "content-addressed store and return the commit hash. Requires agentvfs "
        "write capability. With pair_libos=true it also records a libOS "
        "checkpoint of this process embedding the agentvfs commit hash, giving "
        "one call two-plane checkpoint semantics."
    )
    args_schema = AgentvfsCheckpointArgs
    output_schema = AgentvfsCheckpointOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"agentvfs.write", "checkpoint.write"},
        timeout_s=None,
    )
    tags = ["agentvfs", "checkpoint", "side_effect"]

    def run(self, args: AgentvfsCheckpointArgs, ctx: ToolContext) -> AgentvfsCheckpointOutput:
        adapter = _adapter(ctx)
        label = _validate_label(args.label, "label")
        if not args.pair_libos:
            response = adapter.checkpoint(ctx.pid, label)
            return AgentvfsCheckpointOutput(commit=str(response["commit"]), label=label)
        runtime = _runtime(ctx)
        _require_libos_checkpoint_right(runtime, ctx.pid)
        commit = str(adapter.checkpoint(ctx.pid, label)["commit"])
        checkpoint_id = _create_paired_libos_checkpoint(
            runtime,
            ctx.pid,
            adapter.binding.workspace,
            label,
            commit,
            args.reason,
        )
        return AgentvfsCheckpointOutput(
            commit=commit, label=label, libos_checkpoint_id=checkpoint_id
        )


class AgentvfsRollbackTool(SyncAgentTool[AgentvfsRollbackArgs]):
    name = "agentvfs_rollback"
    description = (
        "Roll the process's agentvfs workspace filesystem back to a checkpoint "
        "label or commit hash, restoring deleted and overwritten files. "
        "Destructive: requires the stronger agentvfs admin capability. With "
        "pair_libos=true it validates and attempts to restore the paired libOS "
        "checkpoint (libOS object/SQL state) as well; when the in-tool libOS "
        "restore is not permitted (missing checkpoint admin authority) or is "
        "refused while the scheduler runs, the filesystem rollback still stands "
        "and the result reports libos_restore=pending_host_restore so the Host "
        "can finish the second plane once the process is quiescent."
    )
    args_schema = AgentvfsRollbackArgs
    output_schema = AgentvfsRollbackOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"agentvfs.admin", "checkpoint.restore"},
        timeout_s=None,
    )
    tags = ["agentvfs", "rollback", "high_risk", "side_effect"]

    def run(self, args: AgentvfsRollbackArgs, ctx: ToolContext) -> AgentvfsRollbackOutput:
        adapter = _adapter(ctx)
        target = _validate_label(args.target, "target")
        if not args.pair_libos:
            response = adapter.rollback(ctx.pid, target)
            return AgentvfsRollbackOutput(
                rolled_back_to=str(response["rolled_back_to"]),
                libos_restore=LIBOS_RESTORE_SKIPPED,
            )
        if not args.libos_checkpoint_id:
            raise ValidationError(
                "agentvfs_rollback with pair_libos=true requires libos_checkpoint_id "
                "from a paired agentvfs_checkpoint result"
            )
        runtime = _runtime(ctx)
        paired_metadata = _inspect_paired_checkpoint(
            runtime, ctx.pid, args.libos_checkpoint_id
        )
        rolled_to = str(adapter.rollback(ctx.pid, target)["rolled_back_to"])
        _require_commit_pairing(paired_metadata, args.libos_checkpoint_id, rolled_to)
        restore_status, hint = _restore_libos_paired(
            runtime, ctx.pid, adapter.binding.workspace, args.libos_checkpoint_id
        )
        return AgentvfsRollbackOutput(
            rolled_back_to=rolled_to,
            paired_libos_checkpoint_id=args.libos_checkpoint_id,
            libos_restore=restore_status,
            libos_restore_hint=hint,
        )


def register_module(ctx: Any) -> None:
    for tool in (
        AgentvfsStatusTool(),
        AgentvfsCheckpointTool(),
        AgentvfsRollbackTool(),
    ):
        ctx.register_tool(tool)

    ctx.register_image(
        AgentImage(
            image_id="agentvfs-agent:v0",
            name="agentvfs-agent",
            default_tools=[
                "process_exit",
                "agentvfs_checkpoint",
                "agentvfs_rollback",
                "agentvfs_status",
            ],
            required_capabilities=[
                {"resource": "agentvfs:*", "rights": ["read", "write", "admin"]}
            ],
            metadata={"module": MODULE_ID},
        )
    )
    ctx.add_startup_hook(initialize_agentvfs)
