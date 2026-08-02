from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import re
import threading
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models import (
    CapabilityDecision,
    CapabilityEffect,
    CapabilityRight,
    DataFlowContext,
    DataSink,
    EventType,
    GitCommit,
    GitDiffResult,
    GitErrorCode,
    GitOperationResult,
    GitPatchArtifact,
    GitPath,
    GitPullRequest,
    GitPullRequestReview,
    GitPullRequestStatus,
    GitRef,
    GitRemoteInfo,
    GitRepositoryInfo,
    GitStateToken,
    GitStatusEntry,
    GitStatusKind,
    GitStatusResult,
    GitWorktreeInfo,
    GitReviewDecision,
    ObjectMetadata,
    ObjectRight,
    ObjectType,
    Provenance,
    ResourceUsage,
)
from agent_libos.models.exceptions import (
    CapabilityDenied,
    GitError,
    HumanApprovalRequired,
    ResourceLimitExceeded,
    ValidationError,
)
from agent_libos.ports import AuditPort, EventPort
from agent_libos.sdk import (
    ProtectedOperationEvidence,
    ProtectedOperationInvocation,
    ProtectedOperationSDK,
    ProviderPhase,
)
from agent_libos.substrate import (
    CommandMetrics,
    GitCommandResult,
    GitProvider,
    GitProviderEffectNotStarted,
    GitRepositoryState,
    GitSubprocessScopeProvider,
    SubprocessLimitExceeded,
    SubprocessLimits,
)
from agent_libos.utils.ids import new_id, utc_now

_T = TypeVar("_T")
_GitMutationCallback = Callable[
    [GitRepositoryState],
    tuple[str | None, dict[str, Any], Sequence[bytes]],
]
_OID_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_BRANCH_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,254}\Z")
_REF_RE = re.compile(r"refs/(?:heads|tags|remotes|agent-libos)/[A-Za-z0-9][A-Za-z0-9._/-]{0,500}\Z")
_REMOTE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_WORKTREE_ID_RE = re.compile(r"(?:main|wt_[A-Za-z0-9_-]{1,96})\Z")
_PR_ID_RE = re.compile(r"pr_[A-Za-z0-9_-]{1,96}\Z")
_GIT_ERROR_PATTERNS: tuple[tuple[GitErrorCode, tuple[bytes, ...]], ...] = (
    (GitErrorCode.NOT_REPOSITORY, (b"not a git repository",)),
    (
        GitErrorCode.AUTH_REQUIRED,
        (
            b"authentication failed",
            b"could not read username",
            b"permission denied (publickey)",
        ),
    ),
    (GitErrorCode.NON_FAST_FORWARD, (b"non-fast-forward", b"fetch first")),
    (GitErrorCode.REMOTE_REJECTED, (b"remote rejected", b"rejected")),
    (GitErrorCode.CONFLICT, (b"conflict", b"unmerged")),
    (
        GitErrorCode.IDENTITY_MISSING,
        (b"user.name", b"user.email", b"author identity unknown"),
    ),
    (GitErrorCode.INVALID_PATH, (b"pathspec", b"outside repository")),
    (
        GitErrorCode.INVALID_REF,
        (b"unknown revision", b"bad revision", b"ambiguous argument"),
    ),
    (
        GitErrorCode.DIRTY_WORKTREE,
        (
            b"would be overwritten",
            b"local changes",
            b"contains modified or untracked files",
        ),
    ),
)
_GIT_MUTATION_DESCRIPTORS = frozenset(
    {
        "primitive.git.mutate",
        "primitive.git.fetch",
        "primitive.git.push",
        "primitive.git.pull_request",
    }
)


@dataclass(frozen=True, slots=True)
class _GitCleanSnapshot:
    flags: str
    preview_sha256: str
    preview_bytes: int
    candidate_paths: tuple[bytes, ...]
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class _GitFlowSnapshot:
    context: DataFlowContext
    state_version: str
    state_version_resolver: Callable[[], str]


@dataclass(frozen=True, slots=True)
class _GitReadFlowSnapshot:
    flow: _GitFlowSnapshot
    repository_state_token: str
    refresh: Callable[[], _GitFlowSnapshot]


_GitFlowSnapshotResolver = Callable[
    [Sequence[tuple[bytes | None, bool]]],
    _GitFlowSnapshot,
]

_GitReadFlowSnapshotResolver = Callable[[], _GitReadFlowSnapshot]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _with_provider_subprocess_scope(
    method: Callable[..., _T],
) -> Callable[..., _T]:
    @wraps(method)
    def scoped(
        primitive: Any,
        pid: str,
        operation: str,
        *args: Any,
        **kwargs: Any,
    ) -> _T:
        with primitive._provider_subprocess_scope(
            pid,
            operation,
            remote=kwargs.get("remote"),
        ):
            return method(primitive, pid, operation, *args, **kwargs)

    return scoped


class GitPrimitive:
    """Capability-controlled, typed Git boundary for the Runtime workspace."""

    def __init__(
        self,
        capabilities: CapabilityManager,
        audit: AuditPort,
        events: EventPort,
        *,
        protected_operations: ProtectedOperationSDK,
        human: Any | None,
        provider: GitProvider,
        filesystem: Any,
        memory: Any | None = None,
        data_flow: Any | None = None,
        config: AgentLibOSConfig | None = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.capabilities = capabilities
        self.audit = audit
        self.events = events
        self.protected_operations = protected_operations
        self.resources = protected_operations.resources
        self.human = human
        self.provider = provider
        self.filesystem = filesystem
        self.memory = memory
        self.data_flow = data_flow
        self._dispatch_state = threading.local()
        self._resource_scope_state = threading.local()

    @property
    def repository_resource(self) -> str:
        return self.config.git.repository_resource

    @staticmethod
    def remote_resource(remote: str) -> str:
        return f"git_remote:workspace:{remote}"

    @staticmethod
    def pull_request_resource(pr_id: str) -> str:
        return f"git_pr:workspace:{pr_id}"

    def _worktree_path(self, worktree_id: str) -> str | Path | None:
        if not isinstance(worktree_id, str) or not _WORKTREE_ID_RE.fullmatch(worktree_id):
            raise GitError(GitErrorCode.INVALID_PATH.value, "invalid managed worktree id")
        if worktree_id == "main":
            return None
        root = getattr(self.provider, "managed_worktree_root", None)
        if root is None:
            raise GitError(GitErrorCode.UNSUPPORTED.value, "Git provider does not support managed worktrees")
        return Path(root) / worktree_id

    @staticmethod
    def _validate_ref_name(value: str, *, branch_only: bool = False) -> str:
        if not isinstance(value, str) or not value or "\x00" in value:
            raise GitError(GitErrorCode.INVALID_REF.value, "invalid Git ref")
        pattern = _BRANCH_RE if branch_only else _REF_RE
        if not pattern.fullmatch(value):
            raise GitError(GitErrorCode.INVALID_REF.value, "invalid Git ref")
        if (
            value.startswith("-")
            or value.endswith(("/", ".", ".lock"))
            or "//" in value
            or ".." in value
            or "@{" in value
            or any(character in value for character in " ~^:?*[\\")
            or any(
                part.startswith(".") or part.endswith(".lock")
                for part in value.split("/")
            )
            or (branch_only and value.casefold() == "head")
        ):
            raise GitError(GitErrorCode.INVALID_REF.value, "invalid Git ref")
        return value

    @classmethod
    def _validate_branch(cls, value: str) -> str:
        return cls._validate_ref_name(value, branch_only=True)

    @staticmethod
    def _validate_remote(remote: str) -> str:
        if not isinstance(remote, str) or not _REMOTE_RE.fullmatch(remote) or remote.startswith("-"):
            raise GitError(GitErrorCode.INVALID_REF.value, "invalid Git remote name")
        return remote

    @staticmethod
    def _validate_expected_token(value: str) -> str:
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise GitError(GitErrorCode.STALE_STATE.value, "expected_state_token is invalid")
        return value

    @staticmethod
    def _stable_generated_id(prefix: str, *parts: str | None) -> str:
        material = json.dumps(
            [prefix, *parts],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"{prefix}_{_sha256(material)[:32]}"

    @staticmethod
    def _decode_path(value: str | GitPath | dict[str, Any]) -> bytes:
        if isinstance(value, GitPath):
            token = value.path_b64
            try:
                raw = base64.b64decode(token, validate=True)
            except (ValueError, TypeError) as exc:
                raise GitError(GitErrorCode.INVALID_PATH.value, "invalid Git path token") from exc
        elif isinstance(value, dict):
            token = value.get("path_b64")
            display = value.get("path") or value.get("display")
            if token is not None:
                try:
                    raw = base64.b64decode(str(token), validate=True)
                except (ValueError, TypeError) as exc:
                    raise GitError(GitErrorCode.INVALID_PATH.value, "invalid Git path token") from exc
            elif isinstance(display, str):
                raw = os.fsencode(display)
            else:
                raise GitError(GitErrorCode.INVALID_PATH.value, "Git path is missing")
        elif isinstance(value, str):
            raw = os.fsencode(value)
        else:
            raise GitError(GitErrorCode.INVALID_PATH.value, "invalid Git path")
        if not raw or b"\x00" in raw or raw.startswith((b"/", b"\\")):
            raise GitError(GitErrorCode.INVALID_PATH.value, "Git path must be workspace-relative")
        normalized = raw.replace(b"\\", b"/") if os.name == "nt" else raw
        parts = normalized.split(b"/")
        if any(part in {b"", b".", b".."} for part in parts):
            raise GitError(GitErrorCode.INVALID_PATH.value, "Git path contains an unsafe component")
        if any(part.lower() == b".git" for part in parts):
            raise GitError(GitErrorCode.INVALID_PATH.value, "Git metadata paths are not valid worktree paths")
        return b"/".join(parts)

    @classmethod
    def _decode_paths(
        cls,
        values: Iterable[str | GitPath | dict[str, Any]],
        *,
        required: bool = False,
    ) -> tuple[bytes, ...]:
        paths = tuple(dict.fromkeys(cls._decode_path(value) for value in values))
        if required and not paths:
            raise GitError(GitErrorCode.INVALID_PATH.value, "at least one Git path is required")
        return paths

    @staticmethod
    def _git_path(raw: bytes) -> GitPath:
        try:
            display = raw.decode("utf-8", errors="strict")
            lossy = False
        except UnicodeDecodeError:
            display = raw.decode("utf-8", errors="replace")
            lossy = True
        return GitPath(
            display=display,
            path_b64=base64.b64encode(raw).decode("ascii"),
            lossy=lossy,
        )

    @staticmethod
    def _path_argv(paths: Sequence[bytes]) -> list[str]:
        return [os.fsdecode(path) for path in paths]

    def _bounded_git_paths(self, raw: bytes) -> list[GitPath]:
        paths = [path for path in raw.split(b"\0") if path]
        if len(paths) > self.config.git.status_entry_hard_limit:
            raise GitError(
                GitErrorCode.OUTPUT_TOO_LARGE.value,
                "Git path output exceeds the configured entry hard limit",
            )
        return [self._git_path(path) for path in paths]

    @staticmethod
    def _state_token(state: GitRepositoryState) -> GitStateToken:
        fields = (
            state.layout.repository_id,
            state.layout.worktree_id,
            state.head_ref or "",
            state.head_oid or "",
            state.index_sha256,
            state.config_sha256,
            state.refs_sha256,
            state.worktrees_sha256,
            state.pull_requests_sha256,
            state.worktree_sha256,
            state.status_sha256,
        )
        digest = hashlib.sha256()
        for field in fields:
            encoded = field.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return GitStateToken(
            token=digest.hexdigest(),
            repository_identity=state.layout.repository_id,
            worktree_id=state.layout.worktree_id,
            head_ref=state.head_ref,
            head_oid=state.head_oid,
            index_sha256=state.index_sha256,
            config_sha256=state.config_sha256,
            refs_sha256=state.refs_sha256,
            worktrees_sha256=state.worktrees_sha256,
            pull_requests_sha256=state.pull_requests_sha256,
            worktree_sha256=state.worktree_sha256,
        )

    @classmethod
    def _require_same_state(cls, expected: str, state: GitRepositoryState) -> GitStateToken:
        selected = cls._state_token(state)
        if selected.token != expected:
            raise GitError(
                GitErrorCode.STALE_STATE.value,
                "Git repository state changed after the preceding read",
                retryable=True,
                details={
                    "actual_state_token": selected.token,
                    "effect_started": False,
                },
            )
        return selected

    def _selected_capability_has_approval_binding(self, decision: CapabilityDecision) -> bool:
        cap_id = decision.selected_capability_id
        if cap_id is None:
            return False
        capability = self.capabilities.store.get_capability(cap_id)
        return bool(
            capability is not None
            and CapabilityManager.APPROVAL_BINDING_KEY in capability.constraints
        )

    def _request_approval(
        self,
        *,
        pid: str,
        resource: str,
        right: CapabilityRight,
        context: dict[str, Any],
        question: str,
        source_oids: Iterable[str] | None = None,
    ) -> None:
        if self.human is None:
            raise CapabilityDenied(f"{pid} requires human approval for {right.value} on {resource}")
        request_id = self.human.query(
            pid=pid,
            human=self.config.runtime.default_human,
            request={
                "type": "external_operation_approval",
                "question": question,
                "requested_once_capability": {
                    "subject": pid,
                    "resource": resource,
                    "rights": [right.value],
                    "constraints": {},
                },
                "context": context,
            },
            blocking=True,
            source_oids=source_oids,
        )
        raise HumanApprovalRequired(
            request_id=request_id,
            message=f"{pid} is waiting for per-use human approval for {resource}",
        )

    def _authorize(
        self,
        *,
        pid: str,
        resource: str,
        right: CapabilityRight,
        context: dict[str, Any],
        question: str,
        mandatory_approval: bool = False,
        source_oids: Iterable[str] | None = None,
    ) -> CapabilityDecision:
        decision = self.capabilities.authorize(pid, resource, right, context, audit=True)
        if decision.allowed:
            if mandatory_approval and not self._selected_capability_has_approval_binding(decision):
                self._request_approval(
                    pid=pid,
                    resource=resource,
                    right=right,
                    context=context,
                    question=question,
                    source_oids=source_oids,
                )
            return decision
        if decision.effect is CapabilityEffect.ASK:
            self._request_approval(
                pid=pid,
                resource=resource,
                right=right,
                context=context,
                question=question,
                source_oids=source_oids,
            )
        raise CapabilityDenied(decision.reason)

    def _operation_context(
        self,
        *,
        pid: str,
        operation: str,
        resource: str,
        right: CapabilityRight,
        worktree_id: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "adapter": "git",
            "primitive": f"runtime.git.{operation}",
            "operation": operation,
            "authority_operation": f"git.{right.value}",
            "pid": pid,
            "resource": resource,
            "right": right.value,
            "worktree_id": worktree_id,
            **dict(extra or {}),
        }

    def _evidence(
        self,
        *,
        pid: str,
        operation: str,
        resource: str,
        summary: dict[str, Any],
        mutation: bool,
        input_refs: Iterable[str] = (),
        output_refs: Iterable[str] = (),
    ) -> ProtectedOperationEvidence:
        event_type = EventType.EXTERNAL_WRITE if mutation else EventType.EXTERNAL_READ
        payload = {"adapter": "git", "operation": operation, **summary}
        return ProtectedOperationEvidence(
            event_type=event_type,
            event_source=pid,
            event_target=resource,
            event_payload=payload,
            audit_action=f"primitive.git.{operation}",
            audit_actor=pid,
            audit_target=resource,
            audit_decision=summary,
            input_refs=tuple(input_refs),
            output_refs=tuple(output_refs),
            effect_metadata=summary,
        )

    @_with_provider_subprocess_scope
    def _read(
        self,
        pid: str,
        operation: str,
        callback: Callable[[], tuple[_T, dict[str, Any]]],
        *,
        right: CapabilityRight = CapabilityRight.READ,
        worktree_id: str = "main",
        extra: dict[str, Any] | None = None,
        resource: str | None = None,
        source_oids: Iterable[str] | None = None,
        flow_snapshot_resolver: _GitReadFlowSnapshotResolver | None = None,
        flow_lock_paths: Sequence[str] = (),
    ) -> _T:
        target = resource or self.repository_resource
        context = self._operation_context(
            pid=pid,
            operation=operation,
            resource=target,
            right=right,
            worktree_id=worktree_id,
            extra=extra,
        )
        decision = self._authorize(
            pid=pid,
            resource=target,
            right=right,
            context=context,
            question=f"Allow this process to inspect Git {operation.replace('_', ' ')}?",
            source_oids=source_oids,
        )
        self.provider.validate_read_only_operation(
            "status",
            worktree=self._worktree_path(worktree_id),
        )
        observation = {"read_only": True, "worktree_id": worktree_id, **dict(extra or {})}
        if self.data_flow is None:
            raise ValidationError("Git data-flow manager is not attached")

        def guarded_flow_snapshot() -> _GitReadFlowSnapshot:
            try:
                stable_flow = self._repository_content_flow_snapshot()
                if flow_snapshot_resolver is not None:
                    operation_flow = flow_snapshot_resolver()
                    read_flow_snapshot = _GitReadFlowSnapshot(
                        flow=self._aggregate_flow_snapshots(
                            (stable_flow, operation_flow.flow)
                        ),
                        repository_state_token=operation_flow.repository_state_token,
                        refresh=lambda: self._aggregate_flow_snapshots(
                            (
                                self._repository_content_flow_snapshot(),
                                operation_flow.refresh(),
                            )
                        ),
                    )
                else:
                    state = self.provider.repository_state(
                        worktree=self._worktree_path(worktree_id)
                    )
                    read_flow_snapshot = _GitReadFlowSnapshot(
                        flow=stable_flow,
                        repository_state_token=self._state_token(state).token,
                        refresh=self._repository_content_flow_snapshot,
                    )
                return read_flow_snapshot
            except GitProviderEffectNotStarted:
                raise

        def guarded_callback() -> tuple[_T, dict[str, Any]]:
            try:
                return callback()
            except GitProviderEffectNotStarted:
                raise
            except GitError as exc:
                if exc.code in {
                    GitErrorCode.GIT_UNAVAILABLE.value,
                    GitErrorCode.UNSUPPORTED_GIT_VERSION.value,
                    GitErrorCode.NOT_REPOSITORY.value,
                    GitErrorCode.UNSAFE_REPOSITORY.value,
                    GitErrorCode.UNSAFE_CONFIG.value,
                }:
                    raise self._provider_not_started(exc) from exc
                raise
        request_context = self.data_flow.context_from_source_oids(
            pid,
            source_oids,
        )
        invocation = ProtectedOperationInvocation(
            pid=pid,
            actor=pid,
            target=target,
            decisions=(decision,),
            canonical_args=context,
            observation=observation,
            data_flow_ingress_context=self.data_flow.unclassified_ingress_context(
                request_context,
                origin="external:git",
            ),
        )
        with self.protected_operations.start(
            "primitive.git.read",
            invocation,
            provider=self.provider,
        ) as protected:
            with self.provider.repository_lock(
                worktree=self._worktree_path(worktree_id)
            ):
                with self.filesystem.hold_file_label_io_paths(flow_lock_paths):
                    read_flow_snapshot = protected.call(
                        ProviderPhase(
                            f"{operation}_flow_snapshot",
                            information_flow=True,
                        ),
                        guarded_flow_snapshot,
                    )
                    self.data_flow.observe_ingress(
                        read_flow_snapshot.flow.context
                    )
                    result, summary = protected.call(
                        ProviderPhase(operation, information_flow=True),
                        guarded_callback,
                    )
                    current_flow = read_flow_snapshot.refresh()
                    reported_state_token = summary.get("state_token")
                    if (
                        (
                            reported_state_token is not None
                            and reported_state_token
                            != read_flow_snapshot.repository_state_token
                        )
                        or current_flow.state_version
                        != read_flow_snapshot.flow.state_version
                    ):
                        self.data_flow.observe_ingress(
                            DataFlowContext.aggregate(
                                (
                                    read_flow_snapshot.flow.context,
                                    current_flow.context,
                                )
                            )
                        )
                        raise CapabilityDenied(
                            "Git data-flow carrier changed during read"
                        )
                    return protected.complete(
                        result,
                        self._evidence(
                            pid=pid,
                            operation=operation,
                            resource=target,
                            summary=summary,
                            mutation=False,
                        ),
                        classification_context=observation,
                        classification_result=summary,
                    )

    def _subprocess_limits(self, pid: str) -> SubprocessLimits | None:
        if self.resources is None:
            return None
        wall = self.resources.remaining_cumulative(
            pid,
            "max_subprocess_wall_seconds",
            "subprocess_wall_seconds",
        )
        cpu = self.resources.remaining_cumulative(
            pid,
            "max_subprocess_cpu_seconds",
            "subprocess_cpu_seconds",
        )
        memory = self.resources.peak_limit(pid, "max_subprocess_memory_bytes")
        if wall is not None and wall <= 0:
            raise ResourceLimitExceeded(
                f"process {pid} exhausted subprocess wall-time budget"
            )
        if cpu is not None and cpu <= 0:
            raise ResourceLimitExceeded(
                f"process {pid} exhausted subprocess CPU budget"
            )
        if memory is not None and memory <= 0:
            raise ResourceLimitExceeded(
                f"process {pid} exhausted subprocess memory budget"
            )
        if wall is None and cpu is None and memory is None:
            return None
        return SubprocessLimits(
            wall_seconds=wall,
            cpu_seconds=cpu,
            memory_bytes=memory,
        )

    @staticmethod
    def _metrics_json(metrics: CommandMetrics | None) -> dict[str, Any] | None:
        if metrics is None:
            return None
        return {
            "wall_seconds": metrics.wall_seconds,
            "cpu_seconds": metrics.cpu_seconds,
            "peak_memory_bytes": metrics.peak_memory_bytes,
            "killed": metrics.killed,
            "limit_kind": metrics.limit_kind,
        }

    def _charge_subprocess_metrics(
        self,
        pid: str,
        metrics: CommandMetrics | None,
        *,
        operation: str,
        remote: str | None,
    ) -> None:
        if self.resources is None or metrics is None:
            return
        if (
            metrics.wall_seconds <= 0
            and metrics.cpu_seconds <= 0
            and metrics.peak_memory_bytes <= 0
        ):
            return
        self.resources.charge(
            pid,
            ResourceUsage(
                subprocess_wall_seconds=max(0.0, metrics.wall_seconds),
                subprocess_cpu_seconds=max(0.0, metrics.cpu_seconds),
                subprocess_peak_memory_bytes=max(0, metrics.peak_memory_bytes),
            ),
            source=f"primitive.git.{operation}",
            context={
                "operation": operation,
                "remote": remote,
                "metrics": self._metrics_json(metrics),
            },
            allow_overage=True,
            kill_on_exceed=True,
        )

    @contextlib.contextmanager
    def _provider_subprocess_scope(
        self,
        pid: str,
        operation: str,
        *,
        remote: str | None = None,
    ) -> Iterator[None]:
        active = getattr(self._resource_scope_state, "current", None)
        if isinstance(active, dict):
            if active.get("pid") != pid:
                raise ValidationError(
                    "nested Git provider subprocess scopes must keep one process owner"
                )
            yield
            return
        limits = self._subprocess_limits(pid)
        scope_provider = (
            self.provider
            if isinstance(self.provider, GitSubprocessScopeProvider)
            else None
        )
        if limits is not None and (
            scope_provider is None
            or not scope_provider.supports_subprocess_limits
        ):
            raise ValidationError(
                "Git provider must support SubprocessLimits for budgeted execution"
            )
        scope: Any | None = None
        limit_error: SubprocessLimitExceeded | None = None
        selected_metrics: CommandMetrics | None = None
        selected_scope = (
            scope_provider.subprocess_scope(limits=limits)
            if scope_provider is not None
            else contextlib.nullcontext(None)
        )
        self._resource_scope_state.current = {"pid": pid}
        try:
            with selected_scope as scope:
                try:
                    yield
                except SubprocessLimitExceeded as exc:
                    limit_error = exc
        finally:
            metrics = getattr(scope, "metrics", None)
            if not isinstance(metrics, CommandMetrics) and limit_error is not None:
                metrics = limit_error.metrics
            selected_metrics = metrics if isinstance(metrics, CommandMetrics) else None
            with contextlib.suppress(AttributeError):
                del self._resource_scope_state.current
            self._charge_subprocess_metrics(
                pid,
                selected_metrics,
                operation=operation,
                remote=remote,
            )
        if limit_error is not None:
            reason = str(limit_error)
            if self.resources is not None:
                self.resources.kill_if_exceeded(
                    pid,
                    reason=reason,
                    limit={
                        "kind": limit_error.metrics.limit_kind,
                        "metrics": self._metrics_json(
                            selected_metrics or limit_error.metrics
                        ),
                    },
                )
            raise ResourceLimitExceeded(reason) from limit_error

    def _run(
        self,
        args: Sequence[str],
        *,
        worktree_id: str,
        max_output_bytes: int | None = None,
        remote: str | None = None,
        expected_remote_fingerprint: str | None = None,
        read_only: bool = True,
        stdin: bytes | None = None,
        verify_after: bool = True,
    ) -> GitCommandResult:
        return self.provider.run(
            args,
            worktree=self._worktree_path(worktree_id),
            max_output_bytes=max_output_bytes,
            remote=remote,
            expected_remote_fingerprint=expected_remote_fingerprint,
            read_only=read_only,
            stdin=stdin,
            verify_after=verify_after,
        )

    @staticmethod
    def _error_from_result(
        result: GitCommandResult,
        operation: str,
        *,
        effect: str = "none",
    ) -> GitError:
        lowered = result.stderr.lower() + b"\n" + result.stdout.lower()
        code = GitErrorCode.COMMAND_FAILED
        for selected_code, patterns in _GIT_ERROR_PATTERNS:
            if any(pattern in lowered for pattern in patterns):
                code = selected_code
                break
        return GitError(
            code.value,
            f"Git {operation} failed",
            operation=operation,
            retryable=False,
            details={
                "effect": effect,
                "stdout_sha256": result.stdout_sha256,
                "stderr_sha256": result.stderr_sha256,
            },
        )

    @classmethod
    def _require_success(
        cls,
        result: GitCommandResult,
        operation: str,
        *,
        effect: str = "none",
    ) -> GitCommandResult:
        if result.returncode != 0:
            raise cls._error_from_result(result, operation, effect=effect)
        return result

    def _resolve_commit(self, value: str, *, worktree_id: str) -> str:
        layout = self.provider.repository_layout(worktree=self._worktree_path(worktree_id))
        if _OID_RE.fullmatch(value):
            expected_length = 40 if layout.object_format == "sha1" else 64
            if len(value) != expected_length:
                raise GitError(GitErrorCode.INVALID_REF.value, "OID length does not match repository object format")
            selected = value
        elif value.startswith("refs/"):
            selected = self._validate_ref_name(value)
        else:
            branch = self._validate_branch(value)
            selected = f"refs/heads/{branch}"
        result = self._run(
            ["rev-parse", "--verify", "--end-of-options", f"{selected}^{{commit}}"],
            worktree_id=worktree_id,
            max_output_bytes=65536,
        )
        self._require_success(result, "resolve_ref")
        oid = result.stdout.strip().decode("ascii", errors="strict")
        if not _OID_RE.fullmatch(oid):
            raise GitError(GitErrorCode.INVALID_REF.value, "Git returned an invalid object id")
        return oid

    def _resolve_ref_oid(self, value: str, *, worktree_id: str) -> str:
        selected = self._validate_ref_name(value)
        result = self._run(
            ["rev-parse", "--verify", "--end-of-options", selected],
            worktree_id=worktree_id,
            max_output_bytes=65536,
        )
        self._require_success(result, "resolve_ref")
        oid = result.stdout.strip().decode("ascii", errors="strict")
        if not _OID_RE.fullmatch(oid):
            raise GitError(GitErrorCode.INVALID_REF.value, "Git returned an invalid object id")
        return oid

    @classmethod
    def _parse_status(
        cls,
        state: GitRepositoryState,
        *,
        limit: int,
    ) -> GitStatusResult:
        entries: list[GitStatusEntry] = []
        branch: str | None = None
        upstream: str | None = None
        ahead = 0
        behind = 0
        records = state.status_porcelain.split(b"\0")
        offset = 0
        total_entries = 0
        while offset < len(records):
            record = records[offset]
            offset += 1
            if not record:
                continue
            if record.startswith(b"# branch.head "):
                value = record[len(b"# branch.head ") :]
                branch = None if value == b"(detached)" else value.decode("utf-8", errors="replace")
                continue
            if record.startswith(b"# branch.upstream "):
                upstream = record[len(b"# branch.upstream ") :].decode("utf-8", errors="replace")
                continue
            if record.startswith(b"# branch.ab "):
                match = re.fullmatch(rb"\+(\d+) -(\d+)", record[len(b"# branch.ab ") :])
                if match:
                    ahead, behind = int(match.group(1)), int(match.group(2))
                continue
            entry: GitStatusEntry | None = None
            if record.startswith(b"1 "):
                parts = record.split(b" ", 8)
                if len(parts) != 9:
                    raise GitError(GitErrorCode.COMMAND_FAILED.value, "invalid porcelain v2 tracked record")
                xy = parts[1].decode("ascii")
                entry = GitStatusEntry(
                    path=cls._git_path(parts[8]),
                    kind=GitStatusKind.TRACKED,
                    index_status=xy[0],
                    worktree_status=xy[1],
                    submodule=parts[2].decode("ascii"),
                    head_mode=parts[3].decode("ascii"),
                    index_mode=parts[4].decode("ascii"),
                    worktree_mode=parts[5].decode("ascii"),
                    head_oid=parts[6].decode("ascii"),
                    index_oid=parts[7].decode("ascii"),
                )
            elif record.startswith(b"2 "):
                parts = record.split(b" ", 9)
                if len(parts) != 10 or offset >= len(records):
                    raise GitError(GitErrorCode.COMMAND_FAILED.value, "invalid porcelain v2 rename record")
                original = records[offset]
                offset += 1
                xy = parts[1].decode("ascii")
                entry = GitStatusEntry(
                    path=cls._git_path(parts[9]),
                    original_path=cls._git_path(original),
                    kind=GitStatusKind.RENAMED,
                    index_status=xy[0],
                    worktree_status=xy[1],
                    submodule=parts[2].decode("ascii"),
                    head_mode=parts[3].decode("ascii"),
                    index_mode=parts[4].decode("ascii"),
                    worktree_mode=parts[5].decode("ascii"),
                    head_oid=parts[6].decode("ascii"),
                    index_oid=parts[7].decode("ascii"),
                )
            elif record.startswith(b"u "):
                parts = record.split(b" ", 10)
                if len(parts) != 11:
                    raise GitError(GitErrorCode.COMMAND_FAILED.value, "invalid porcelain v2 unmerged record")
                xy = parts[1].decode("ascii")
                entry = GitStatusEntry(
                    path=cls._git_path(parts[10]),
                    kind=GitStatusKind.UNMERGED,
                    index_status=xy[0],
                    worktree_status=xy[1],
                    submodule=parts[2].decode("ascii"),
                    head_mode=parts[3].decode("ascii"),
                    index_mode=parts[4].decode("ascii"),
                    worktree_mode=parts[6].decode("ascii"),
                    head_oid=parts[7].decode("ascii"),
                    index_oid=parts[8].decode("ascii"),
                )
            elif record.startswith(b"? "):
                entry = GitStatusEntry(
                    path=cls._git_path(record[2:]),
                    kind=GitStatusKind.UNTRACKED,
                    index_status="?",
                    worktree_status="?",
                )
            elif record.startswith(b"! "):
                entry = GitStatusEntry(
                    path=cls._git_path(record[2:]),
                    kind=GitStatusKind.IGNORED,
                    index_status="!",
                    worktree_status="!",
                )
            elif not record.startswith(b"# "):
                raise GitError(GitErrorCode.COMMAND_FAILED.value, "unsupported porcelain v2 record")
            if entry is not None:
                total_entries += 1
                if len(entries) < limit:
                    entries.append(entry)
        return cls._status_result(
            state,
            branch=branch,
            upstream=upstream,
            ahead=ahead,
            behind=behind,
            entries=entries,
            total_entries=total_entries,
            limit=limit,
        )

    @classmethod
    def _status_result(
        cls,
        state: GitRepositoryState,
        *,
        branch: str | None,
        upstream: str | None,
        ahead: int,
        behind: int,
        entries: list[GitStatusEntry],
        total_entries: int,
        limit: int,
    ) -> GitStatusResult:
        return GitStatusResult(
            repository_id=state.layout.repository_id,
            worktree_id=state.layout.worktree_id,
            branch=branch,
            upstream=upstream,
            ahead=ahead,
            behind=behind,
            head_oid=state.head_oid,
            entries=entries,
            state=cls._state_token(state),
            truncated=total_entries > limit,
            bytes=len(state.status_porcelain),
            sha256=state.status_sha256,
        )

    def repository_info(self, pid: str, *, worktree_id: str = "main") -> GitRepositoryInfo:
        worktree = self._worktree_path(worktree_id)

        def read() -> tuple[GitRepositoryInfo, dict[str, Any]]:
            state = self.provider.repository_state(worktree=worktree)
            token = self._state_token(state)
            result = GitRepositoryInfo(
                repository_id=state.layout.repository_id,
                worktree_id=state.layout.worktree_id,
                root=str(state.layout.root),
                git_dir=str(state.layout.git_dir),
                common_dir=str(state.layout.common_dir),
                object_format=state.layout.object_format,
                bare=False,
                linked_worktree=state.layout.linked_worktree,
                git_version=state.layout.git_version,
                state=token,
            )
            return result, {
                "repository_id": result.repository_id,
                "worktree_id": result.worktree_id,
                "object_format": result.object_format,
                "state_token": token.token,
            }

        return self._read(pid, "repository_info", read, worktree_id=worktree_id)

    def status(
        self,
        pid: str,
        *,
        worktree_id: str = "main",
        limit: int | None = None,
    ) -> GitStatusResult:
        selected_limit = self.config.git.status_entry_limit if limit is None else int(limit)
        if selected_limit <= 0 or selected_limit > self.config.git.status_entry_hard_limit:
            raise ValidationError("Git status limit is outside the configured bounds")
        worktree = self._worktree_path(worktree_id)

        def read() -> tuple[GitStatusResult, dict[str, Any]]:
            state = self.provider.repository_state(worktree=worktree)
            result = self._parse_status(state, limit=selected_limit)
            return result, {
                "repository_id": result.repository_id,
                "worktree_id": result.worktree_id,
                "entries": len(result.entries),
                "truncated": result.truncated,
                "bytes": result.bytes,
                "sha256": result.sha256,
                "state_token": result.state.token,
            }

        return self._read(
            pid,
            "status",
            read,
            worktree_id=worktree_id,
            extra={"limit": selected_limit},
            flow_snapshot_resolver=lambda: self._git_read_lineage_snapshot(
                worktree_id=worktree_id,
                include_head=True,
                include_index=True,
                include_repository_content=True,
                include_worktree_tree=True,
            ),
            flow_lock_paths=(self._normalized_worktree_root(worktree_id) or ".",),
        )

    def _bounded_output_limit(self, value: int | None, *, patch: bool = False) -> int:
        default = self.config.git.patch_max_bytes if patch else self.config.git.output_max_bytes
        hard = self.config.git.patch_hard_limit_bytes if patch else self.config.git.output_hard_limit_bytes
        selected = default if value is None else int(value)
        if selected <= 0 or selected > hard:
            raise GitError(GitErrorCode.OUTPUT_TOO_LARGE.value, "requested Git output limit exceeds its hard bound")
        return selected

    def _diff_arguments(
        self,
        *,
        scope: str,
        base: str | None,
        head: str | None,
        paths: Sequence[bytes],
        worktree_id: str,
        name_only: bool,
    ) -> tuple[list[str], str | None, str | None]:
        args = ["diff", "--no-renames", "--no-ext-diff", "--no-textconv"]
        if not name_only:
            args.append("--binary")
        else:
            args.extend(["--name-only", "-z"])
        base_oid: str | None = None
        head_oid: str | None = None
        if scope == "worktree":
            if base is not None or head is not None:
                raise GitError(GitErrorCode.INVALID_REF.value, "worktree diff does not accept base/head")
        elif scope == "staged":
            if base is not None or head is not None:
                raise GitError(GitErrorCode.INVALID_REF.value, "staged diff does not accept base/head")
            args.append("--cached")
        elif scope == "range":
            if base is None or head is None:
                raise GitError(GitErrorCode.INVALID_REF.value, "range diff requires base and head")
            base_oid = self._resolve_commit(base, worktree_id=worktree_id)
            head_oid = self._resolve_commit(head, worktree_id=worktree_id)
            args.extend([base_oid, head_oid])
        else:
            raise ValidationError("Git diff scope must be worktree, staged, or range")
        args.append("--")
        args.extend(self._path_argv(paths))
        return args, base_oid, head_oid

    def diff(
        self,
        pid: str,
        *,
        scope: str = "worktree",
        base: str | None = None,
        head: str | None = None,
        paths: Iterable[str | GitPath | dict[str, Any]] = (),
        worktree_id: str = "main",
        max_bytes: int | None = None,
    ) -> GitDiffResult:
        selected_paths = self._decode_paths(paths)
        selected_limit = self._bounded_output_limit(max_bytes, patch=True)
        hard_limit = self.config.git.patch_hard_limit_bytes

        def read() -> tuple[GitDiffResult, dict[str, Any]]:
            return self._diff_result(
                scope=scope,
                base=base,
                head=head,
                paths=selected_paths,
                worktree_id=worktree_id,
                selected_limit=selected_limit,
                hard_limit=hard_limit,
                operation="diff",
            )

        return self._read(
            pid,
            "diff",
            read,
            right=CapabilityRight.DIFF,
            worktree_id=worktree_id,
            extra={
                "scope": scope,
                "base": base,
                "head": head,
                "path_count": len(selected_paths),
                "paths_sha256": _sha256(b"\0".join(selected_paths)),
                "max_bytes": selected_limit,
            },
            flow_snapshot_resolver=lambda: self._diff_read_flow_snapshot(
                scope=scope,
                base=base,
                head=head,
                paths=selected_paths,
                worktree_id=worktree_id,
            ),
            flow_lock_paths=(
                tuple(
                    self._normalized_file_path(path, worktree_id=worktree_id)
                    for path in selected_paths
                )
                or (self._normalized_worktree_root(worktree_id) or ".",)
                if scope == "worktree"
                else ()
            ),
        )

    def _diff_result(
        self,
        *,
        scope: str,
        base: str | None,
        head: str | None,
        paths: Sequence[bytes],
        worktree_id: str,
        selected_limit: int,
        hard_limit: int,
        operation: str,
    ) -> tuple[GitDiffResult, dict[str, Any]]:
        before = self.provider.repository_state(
            worktree=self._worktree_path(worktree_id)
        )
        args, base_oid, head_oid = self._diff_arguments(
            scope=scope,
            base=base,
            head=head,
            paths=paths,
            worktree_id=worktree_id,
            name_only=False,
        )
        patch_result = self._require_success(
            self._run(
                args,
                worktree_id=worktree_id,
                max_output_bytes=hard_limit,
            ),
            operation,
        )
        name_args, _base_oid, _head_oid = self._diff_arguments(
            scope=scope,
            base=base_oid or base,
            head=head_oid or head,
            paths=paths,
            worktree_id=worktree_id,
            name_only=True,
        )
        names = self._require_success(
            self._run(
                name_args,
                worktree_id=worktree_id,
                max_output_bytes=self.config.git.output_hard_limit_bytes,
            ),
            operation,
        )
        after = self.provider.repository_state(
            worktree=self._worktree_path(worktree_id)
        )
        if self._state_token(before).token != self._state_token(after).token:
            raise GitError(
                GitErrorCode.STALE_STATE.value,
                "Git state changed while diff was being read",
                operation=operation,
                retryable=True,
            )
        full = patch_result.stdout
        returned = full[:selected_limit]
        result = GitDiffResult(
            repository_id=before.layout.repository_id,
            worktree_id=before.layout.worktree_id,
            scope=scope,
            base_oid=base_oid,
            head_oid=head_oid,
            patch=returned.decode("utf-8", errors="replace"),
            patch_b64=base64.b64encode(returned).decode("ascii"),
            changed_paths=self._bounded_git_paths(names.stdout),
            state=self._state_token(before),
            truncated=len(full) > selected_limit,
            bytes=len(full),
            sha256=patch_result.stdout_sha256,
        )
        return result, {
            "repository_id": result.repository_id,
            "worktree_id": result.worktree_id,
            "scope": scope,
            "base_oid": base_oid,
            "head_oid": head_oid,
            "changed_paths": len(result.changed_paths),
            "truncated": result.truncated,
            "bytes": result.bytes,
            "sha256": result.sha256,
            "state_token": result.state.token,
        }

    @staticmethod
    def _parse_commits(raw: bytes, *, limit: int) -> tuple[list[GitCommit], bool]:
        fields = raw.split(b"\0")
        commits: list[GitCommit] = []
        offset = 0
        total = 0
        while offset < len(fields):
            while offset < len(fields) and not _OID_RE.fullmatch(fields[offset].decode("ascii", errors="ignore")):
                offset += 1
            if offset >= len(fields):
                break
            if offset + 9 >= len(fields):
                raise GitError(GitErrorCode.COMMAND_FAILED.value, "Git log returned an incomplete commit record")
            record = fields[offset : offset + 10]
            offset += 10
            oid = record[0].decode("ascii", errors="strict")
            parents = [item for item in record[1].decode("ascii", errors="strict").split(" ") if item]
            total += 1
            if len(commits) < limit:
                commits.append(
                    GitCommit(
                        oid=oid,
                        parents=parents,
                        author_name=record[2].decode("utf-8", errors="replace"),
                        author_email=record[3].decode("utf-8", errors="replace"),
                        authored_at=record[4].decode("ascii", errors="replace"),
                        committer_name=record[5].decode("utf-8", errors="replace"),
                        committer_email=record[6].decode("utf-8", errors="replace"),
                        committed_at=record[7].decode("ascii", errors="replace"),
                        subject=record[8].decode("utf-8", errors="replace"),
                        body=record[9].decode("utf-8", errors="replace"),
                    )
                )
        return commits, total > limit

    @staticmethod
    def _log_format() -> str:
        return "%H%x00%P%x00%an%x00%ae%x00%aI%x00%cn%x00%ce%x00%cI%x00%s%x00%b%x00"

    def log(
        self,
        pid: str,
        *,
        ref: str | None = None,
        limit: int | None = None,
        worktree_id: str = "main",
    ) -> dict[str, Any]:
        selected_limit = self.config.git.log_entry_limit if limit is None else int(limit)
        if selected_limit <= 0 or selected_limit > self.config.git.log_entry_hard_limit:
            raise ValidationError("Git log limit is outside the configured bounds")

        def read() -> tuple[dict[str, Any], dict[str, Any]]:
            before = self.provider.repository_state(worktree=self._worktree_path(worktree_id))
            if ref is None:
                if before.head_oid is None:
                    oid = None
                else:
                    oid = before.head_oid
            else:
                oid = self._resolve_commit(ref, worktree_id=worktree_id)
            if oid is None:
                raw = b""
                commits: list[GitCommit] = []
                truncated = False
            else:
                result = self._require_success(
                    self._run(
                        [
                            "log",
                            "--no-decorate",
                            "--no-show-signature",
                            "-z",
                            f"--max-count={selected_limit + 1}",
                            f"--format={self._log_format()}",
                            oid,
                            "--",
                        ],
                        worktree_id=worktree_id,
                        max_output_bytes=self.config.git.output_hard_limit_bytes,
                    ),
                    "log",
                )
                raw = result.stdout
                commits, truncated = self._parse_commits(raw, limit=selected_limit)
            after = self.provider.repository_state(worktree=self._worktree_path(worktree_id))
            token = self._state_token(before)
            if token.token != self._state_token(after).token:
                raise GitError(GitErrorCode.STALE_STATE.value, "Git state changed while log was being read", retryable=True)
            payload = {
                "commits": commits,
                "truncated": truncated,
                "bytes": len(raw),
                "sha256": _sha256(raw),
                "state": token,
            }
            summary = {
                "repository_id": before.layout.repository_id,
                "worktree_id": before.layout.worktree_id,
                "ref_oid": oid,
                "commits": len(commits),
                "truncated": truncated,
                "bytes": len(raw),
                "sha256": payload["sha256"],
                "state_token": token.token,
            }
            return payload, summary

        return self._read(
            pid,
            "log",
            read,
            worktree_id=worktree_id,
            extra={"ref": ref, "limit": selected_limit},
            flow_snapshot_resolver=lambda: self._log_read_flow_snapshot(
                ref=ref,
                limit=selected_limit,
                worktree_id=worktree_id,
            ),
        )

    def _commit(self, oid: str, *, worktree_id: str) -> GitCommit:
        result = self._require_success(
            self._run(
                [
                    "show",
                    "-s",
                    "--no-show-signature",
                    f"--format={self._log_format()}",
                    oid,
                    "--",
                ],
                worktree_id=worktree_id,
                max_output_bytes=self.config.git.output_hard_limit_bytes,
            ),
            "show",
        )
        commits, _truncated = self._parse_commits(result.stdout, limit=1)
        if len(commits) != 1:
            raise GitError(GitErrorCode.COMMAND_FAILED.value, "Git show did not return one commit")
        return commits[0]

    def show(
        self,
        pid: str,
        ref: str,
        *,
        worktree_id: str = "main",
        max_bytes: int | None = None,
    ) -> dict[str, Any]:
        selected_limit = self._bounded_output_limit(max_bytes, patch=True)

        def read() -> tuple[dict[str, Any], dict[str, Any]]:
            before = self.provider.repository_state(worktree=self._worktree_path(worktree_id))
            oid = self._resolve_commit(ref, worktree_id=worktree_id)
            commit = self._commit(oid, worktree_id=worktree_id)
            parents: tuple[str | None, ...] = tuple(commit.parents) or (None,)
            if len(parents) > self.config.git.log_entry_hard_limit:
                raise GitError(
                    GitErrorCode.OUTPUT_TOO_LARGE.value,
                    "Git commit parent count exceeds the configured hard limit",
                    operation="show",
                )
            parent_diffs: list[dict[str, Any]] = []
            aggregate_patch_bytes = 0
            aggregate_name_bytes = 0
            for parent_oid in parents:
                if parent_oid is None:
                    patch_args = [
                        "show",
                        "--format=",
                        "--no-renames",
                        "--no-ext-diff",
                        "--no-textconv",
                        "--binary",
                        oid,
                        "--",
                    ]
                    names_args = [
                        "diff-tree",
                        "--root",
                        "--no-commit-id",
                        "--name-only",
                        "-r",
                        "-z",
                        oid,
                        "--",
                    ]
                else:
                    patch_args = [
                        "diff",
                        "--no-renames",
                        "--no-ext-diff",
                        "--no-textconv",
                        "--binary",
                        parent_oid,
                        oid,
                        "--",
                    ]
                    names_args = [
                        "diff",
                        "--no-renames",
                        "--no-ext-diff",
                        "--no-textconv",
                        "--name-only",
                        "-z",
                        parent_oid,
                        oid,
                        "--",
                    ]
                patch_result = self._require_success(
                    self._run(
                        patch_args,
                        worktree_id=worktree_id,
                        max_output_bytes=self.config.git.patch_hard_limit_bytes,
                    ),
                    "show",
                )
                aggregate_patch_bytes += len(patch_result.stdout)
                if aggregate_patch_bytes > self.config.git.patch_hard_limit_bytes:
                    raise GitError(
                        GitErrorCode.OUTPUT_TOO_LARGE.value,
                        "aggregate Git parent diffs exceed the configured hard limit",
                        operation="show",
                    )
                names_result = self._require_success(
                    self._run(
                        names_args,
                        worktree_id=worktree_id,
                        max_output_bytes=self.config.git.output_hard_limit_bytes,
                    ),
                    "show",
                )
                aggregate_name_bytes += len(names_result.stdout)
                if aggregate_name_bytes > self.config.git.output_hard_limit_bytes:
                    raise GitError(
                        GitErrorCode.OUTPUT_TOO_LARGE.value,
                        "aggregate Git parent paths exceed the configured hard limit",
                        operation="show",
                    )
                full_parent_patch = patch_result.stdout
                returned_parent_patch = full_parent_patch[:selected_limit]
                parent_diffs.append(
                    {
                        "parent_oid": parent_oid,
                        "patch": returned_parent_patch.decode("utf-8", errors="replace"),
                        "patch_b64": base64.b64encode(returned_parent_patch).decode("ascii"),
                        "changed_paths": self._bounded_git_paths(names_result.stdout),
                        "truncated": len(full_parent_patch) > selected_limit,
                        "bytes": len(full_parent_patch),
                        "sha256": patch_result.stdout_sha256,
                    }
                )
            after = self.provider.repository_state(worktree=self._worktree_path(worktree_id))
            token = self._state_token(before)
            if token.token != self._state_token(after).token:
                raise GitError(GitErrorCode.STALE_STATE.value, "Git state changed while commit was being read", retryable=True)
            canonical = parent_diffs[0]
            payload = {
                "commit": commit,
                # Preserve the historical top-level fields as the root or
                # first-parent view. Multi-parent callers must inspect every
                # entry in parent_diffs before claiming complete evidence.
                "patch_base_oid": canonical["parent_oid"],
                "patch": canonical["patch"],
                "patch_b64": canonical["patch_b64"],
                "changed_paths": canonical["changed_paths"],
                "truncated": canonical["truncated"],
                "bytes": canonical["bytes"],
                "sha256": canonical["sha256"],
                "parent_diffs": parent_diffs,
                "parent_diffs_truncated": any(
                    bool(item["truncated"]) for item in parent_diffs
                ),
                "state": token,
            }
            summary = {
                "repository_id": before.layout.repository_id,
                "worktree_id": before.layout.worktree_id,
                "commit_oid": oid,
                "changed_paths": len(payload["changed_paths"]),
                "truncated": payload["truncated"],
                "bytes": payload["bytes"],
                "sha256": payload["sha256"],
                "parent_diffs": len(parent_diffs),
                "parent_diffs_truncated": payload["parent_diffs_truncated"],
                "state_token": token.token,
            }
            return payload, summary

        return self._read(
            pid,
            "show",
            read,
            worktree_id=worktree_id,
            extra={"ref": ref, "max_bytes": selected_limit},
            flow_snapshot_resolver=lambda: self._show_read_flow_snapshot(
                worktree_id=worktree_id,
                ref=ref,
            ),
        )

    def blame(
        self,
        pid: str,
        path: str | GitPath | dict[str, Any],
        *,
        ref: str | None = None,
        worktree_id: str = "main",
        max_bytes: int | None = None,
    ) -> dict[str, Any]:
        selected_path = self._decode_path(path)
        selected_limit = self._bounded_output_limit(max_bytes)

        def read() -> tuple[dict[str, Any], dict[str, Any]]:
            before = self.provider.repository_state(worktree=self._worktree_path(worktree_id))
            args = ["blame", "--porcelain", "--no-progress"]
            oid = self._resolve_commit(ref, worktree_id=worktree_id) if ref is not None else before.head_oid
            if oid is not None:
                args.append(oid)
            args.extend(["--", os.fsdecode(selected_path)])
            result = self._require_success(
                self._run(args, worktree_id=worktree_id, max_output_bytes=self.config.git.output_hard_limit_bytes),
                "blame",
            )
            after = self.provider.repository_state(worktree=self._worktree_path(worktree_id))
            token = self._state_token(before)
            if token.token != self._state_token(after).token:
                raise GitError(GitErrorCode.STALE_STATE.value, "Git state changed while blame was being read", retryable=True)
            returned = result.stdout[:selected_limit]
            payload = {
                "path": self._git_path(selected_path),
                "ref_oid": oid,
                "content": returned.decode("utf-8", errors="replace"),
                "content_b64": base64.b64encode(returned).decode("ascii"),
                "truncated": len(result.stdout) > selected_limit,
                "bytes": len(result.stdout),
                "sha256": result.stdout_sha256,
                "state": token,
            }
            summary = {
                "repository_id": before.layout.repository_id,
                "worktree_id": before.layout.worktree_id,
                "path_sha256": _sha256(selected_path),
                "ref_oid": oid,
                "truncated": payload["truncated"],
                "bytes": payload["bytes"],
                "sha256": payload["sha256"],
                "state_token": token.token,
            }
            return payload, summary

        return self._read(
            pid,
            "blame",
            read,
            worktree_id=worktree_id,
            extra={
                "path_sha256": _sha256(selected_path),
                "ref": ref,
                "max_bytes": selected_limit,
            },
            flow_snapshot_resolver=lambda: self._blame_read_flow_snapshot(
                path=selected_path,
                ref=ref,
                worktree_id=worktree_id,
            ),
            flow_lock_paths=(
                self._normalized_file_path(
                    selected_path,
                    worktree_id=worktree_id,
                ),
            ),
        )

    def list_refs(
        self,
        pid: str,
        *,
        kind: str = "all",
        limit: int = 200,
        worktree_id: str = "main",
    ) -> dict[str, Any]:
        if limit <= 0 or limit > self.config.git.status_entry_hard_limit:
            raise ValidationError("Git ref limit is outside the configured bounds")
        prefixes = {
            "all": None,
            "branches": "refs/heads/",
            "tags": "refs/tags/",
            "remotes": "refs/remotes/",
            "pull_requests": "refs/agent-libos/pull-requests/",
        }
        if kind not in prefixes:
            raise ValidationError("Git ref kind is invalid")

        def read() -> tuple[dict[str, Any], dict[str, Any]]:
            before = self.provider.repository_state(worktree=self._worktree_path(worktree_id))
            args = [
                "for-each-ref",
                f"--count={limit + 1}",
                "--format=%(refname)%00%(refname:short)%00%(objectname)%00%(symref)%00%(upstream)%00%(upstream:track)",
            ]
            if prefixes[kind] is not None:
                args.append(prefixes[kind] or "")
            result = self._require_success(
                self._run(args, worktree_id=worktree_id, max_output_bytes=self.config.git.output_hard_limit_bytes),
                "list_refs",
            )
            refs: list[GitRef] = []
            rows = result.stdout.splitlines()
            for row in rows[:limit]:
                fields = row.split(b"\0")
                if len(fields) != 6:
                    raise GitError(GitErrorCode.COMMAND_FAILED.value, "Git ref output is malformed")
                name = fields[0].decode("utf-8", errors="strict")
                track = fields[5].decode("ascii", errors="replace")
                ahead_match = re.search(r"ahead (\d+)", track)
                behind_match = re.search(r"behind (\d+)", track)
                ref_kind = (
                    "branch" if name.startswith("refs/heads/") else
                    "tag" if name.startswith("refs/tags/") else
                    "remote" if name.startswith("refs/remotes/") else
                    "pull_request" if name.startswith("refs/agent-libos/pull-requests/") else
                    "other"
                )
                refs.append(
                    GitRef(
                        name=name,
                        short_name=fields[1].decode("utf-8", errors="strict"),
                        kind=ref_kind,
                        oid=fields[2].decode("ascii", errors="strict"),
                        symbolic_target=fields[3].decode("utf-8", errors="strict") or None,
                        upstream=fields[4].decode("utf-8", errors="strict") or None,
                        ahead=int(ahead_match.group(1)) if ahead_match else None,
                        behind=int(behind_match.group(1)) if behind_match else None,
                    )
                )
            after = self.provider.repository_state(worktree=self._worktree_path(worktree_id))
            token = self._state_token(before)
            if token.token != self._state_token(after).token:
                raise GitError(GitErrorCode.STALE_STATE.value, "Git refs changed while being listed", retryable=True)
            payload = {
                "refs": refs,
                "truncated": len(rows) > limit,
                "bytes": len(result.stdout),
                "sha256": result.stdout_sha256,
                "state": token,
            }
            summary = {
                "repository_id": before.layout.repository_id,
                "worktree_id": before.layout.worktree_id,
                "kind": kind,
                "refs": len(refs),
                "truncated": payload["truncated"],
                "bytes": payload["bytes"],
                "sha256": payload["sha256"],
                "state_token": token.token,
            }
            return payload, summary

        return self._read(
            pid,
            "list_refs",
            read,
            worktree_id=worktree_id,
            extra={"kind": kind, "limit": limit},
            flow_snapshot_resolver=lambda: self._git_read_lineage_snapshot(
                worktree_id=worktree_id,
                fixed_carriers=(
                    (("pull_request_collection", "workspace"),)
                    if kind in {"all", "pull_requests"}
                    else ()
                ),
            ),
        )

    def list_remotes(self, pid: str, *, worktree_id: str = "main") -> dict[str, Any]:
        def read() -> tuple[dict[str, Any], dict[str, Any]]:
            before = self.provider.repository_state(worktree=self._worktree_path(worktree_id))
            result = self._require_success(
                self._run(["remote"], worktree_id=worktree_id, max_output_bytes=self.config.git.output_hard_limit_bytes),
                "list_remotes",
            )
            remotes: list[GitRemoteInfo] = []
            remote_rows = result.stdout.splitlines()
            if len(remote_rows) > self.config.git.status_entry_hard_limit:
                raise GitError(
                    GitErrorCode.OUTPUT_TOO_LARGE.value,
                    "Git remote count exceeds the configured hard limit",
                )
            for raw in remote_rows:
                name = raw.decode("utf-8", errors="strict")
                self._validate_remote(name)
                _, _, fingerprint = self.provider.remote_configuration(
                    name,
                    worktree=self._worktree_path(worktree_id),
                )
                remotes.append(
                    GitRemoteInfo(
                        name=name,
                        fetch_url=f"<redacted:{fingerprint['fetch_url_sha256'][:16]}>",
                        push_url=f"<redacted:{fingerprint['push_url_sha256'][:16]}>",
                        fetch_url_sha256=fingerprint["fetch_url_sha256"],
                        push_url_sha256=fingerprint["push_url_sha256"],
                        fetch_refspecs=list(fingerprint.get("fetch_refspecs", ())),
                    )
                )
            after = self.provider.repository_state(worktree=self._worktree_path(worktree_id))
            token = self._state_token(before)
            if token.token != self._state_token(after).token:
                raise GitError(GitErrorCode.STALE_STATE.value, "Git remotes changed while being listed", retryable=True)
            payload = {
                "remotes": remotes,
                "truncated": False,
                "bytes": len(result.stdout),
                "sha256": result.stdout_sha256,
                "state": token,
            }
            summary = {
                "repository_id": before.layout.repository_id,
                "worktree_id": before.layout.worktree_id,
                "remotes": len(remotes),
                "truncated": False,
                "bytes": payload["bytes"],
                "sha256": payload["sha256"],
                "state_token": token.token,
            }
            return payload, summary

        return self._read(pid, "list_remotes", read, worktree_id=worktree_id)

    @staticmethod
    def _parse_worktree_record(
        record: bytes,
    ) -> tuple[dict[str, bytes], set[str]]:
        values: dict[str, bytes] = {}
        flags: set[str] = set()
        for field in record.strip(b"\0").split(b"\0"):
            key, separator, value = field.partition(b" ")
            key_text = key.decode("ascii", errors="strict")
            if separator:
                values[key_text] = value
            else:
                flags.add(key_text)
        if "worktree" not in values:
            raise GitError(
                GitErrorCode.COMMAND_FAILED.value,
                "Git worktree output is malformed",
            )
        return values, flags

    def _listed_worktree_identity(
        self,
        path: Path,
        *,
        before: GitRepositoryState,
        managed_root: Path,
    ) -> tuple[str, bool] | None:
        if path == before.layout.root:
            return "main", False
        managed = bool(
            path.parent == managed_root
            and _WORKTREE_ID_RE.fullmatch(path.name)
            and path.name != "main"
        )
        if not managed:
            return None
        layout = self.provider.repository_layout(worktree=path)
        if (
            layout.root != path
            or layout.repository_id != before.layout.repository_id
            or layout.common_dir != before.layout.common_dir
            or not layout.linked_worktree
        ):
            raise GitError(
                GitErrorCode.UNSAFE_REPOSITORY.value,
                "managed Git worktree failed structural validation",
            )
        return path.name, True

    def _listed_worktree_info(
        self,
        record: bytes,
        *,
        before: GitRepositoryState,
        managed_root: Path,
    ) -> GitWorktreeInfo | None:
        values, flags = self._parse_worktree_record(record)
        path = Path(os.path.abspath(Path(os.fsdecode(values["worktree"]))))
        identity = self._listed_worktree_identity(
            path,
            before=before,
            managed_root=managed_root,
        )
        if identity is None:
            return None
        item_id, managed = identity
        if "bare" in flags:
            raise GitError(
                GitErrorCode.UNSAFE_REPOSITORY.value,
                "Git worktree list contains an unsupported bare entry",
            )
        branch_value = values.get("branch")
        return GitWorktreeInfo(
            worktree_id=item_id,
            path=str(path),
            head_oid=values.get("HEAD", b"").decode("ascii", errors="strict") or None,
            branch=(
                branch_value.decode("utf-8", errors="strict")
                if branch_value
                else None
            ),
            detached="detached" in flags,
            bare=False,
            locked="locked" in flags or "locked" in values,
            prunable="prunable" in flags or "prunable" in values,
            managed=managed,
        )

    def list_worktrees(self, pid: str) -> dict[str, Any]:
        def read() -> tuple[dict[str, Any], dict[str, Any]]:
            before = self.provider.repository_state()
            result = self._require_success(
                self._run(
                    ["worktree", "list", "--porcelain", "-z"],
                    worktree_id="main",
                    max_output_bytes=self.config.git.output_hard_limit_bytes,
                ),
                "list_worktrees",
            )
            records = [record for record in result.stdout.split(b"\0\0") if record]
            if len(records) > self.config.git.status_entry_hard_limit:
                raise GitError(
                    GitErrorCode.OUTPUT_TOO_LARGE.value,
                    "Git worktree count exceeds the configured hard limit",
                )
            worktrees: list[GitWorktreeInfo] = []
            managed_root = Path(
                getattr(
                    self.provider,
                    "managed_worktree_root",
                    Path("/__unmanaged__"),
                )
            )
            for record in records:
                item = self._listed_worktree_info(
                    record,
                    before=before,
                    managed_root=managed_root,
                )
                if item is not None:
                    worktrees.append(item)
            after = self.provider.repository_state()
            token = self._state_token(before)
            if token.token != self._state_token(after).token:
                raise GitError(GitErrorCode.STALE_STATE.value, "Git worktrees changed while being listed", retryable=True)
            payload = {
                "worktrees": worktrees,
                "truncated": False,
                "bytes": len(result.stdout),
                "sha256": result.stdout_sha256,
                "state": token,
            }
            summary = {
                "repository_id": before.layout.repository_id,
                "worktrees": len(worktrees),
                "managed_worktrees": sum(1 for item in worktrees if item.managed),
                "truncated": False,
                "bytes": payload["bytes"],
                "sha256": payload["sha256"],
                "state_token": token.token,
            }
            return payload, summary

        return self._read(pid, "list_worktrees", read, worktree_id="main")

    def _filesystem_resource(
        self,
        raw_path: bytes | None,
        *,
        worktree_id: str,
        subtree: bool,
    ) -> str:
        if worktree_id == "main":
            base = ""
        else:
            worktree = Path(self._worktree_path(worktree_id) or "")
            root = Path(getattr(self.provider, "workspace_root", self.filesystem.root))
            try:
                base = worktree.relative_to(root).as_posix()
            except ValueError as exc:
                raise GitError(GitErrorCode.INVALID_PATH.value, "managed worktree escaped the workspace") from exc
        if raw_path is None:
            relative = base
        else:
            suffix = os.fsdecode(raw_path).replace(os.sep, "/")
            relative = f"{base}/{suffix}".strip("/")
        return (
            self.filesystem.directory_resource_for(relative or ".")
            if subtree
            else self.filesystem.resource_for(relative or ".")
        )

    @staticmethod
    def _flow_state_version(items: Sequence[tuple[str, str, str]]) -> str:
        digest = hashlib.sha256()
        for kind, path, version in items:
            for value in (kind, path, version):
                encoded = value.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
        return digest.hexdigest()

    @staticmethod
    def _combined_flow_state_version(
        expected_state_token: str,
        source_state_version: str,
    ) -> str:
        return _sha256(
            expected_state_token.encode("ascii")
            + b"\0"
            + source_state_version.encode("ascii")
        )

    def _filesystem_flow_snapshot(
        self,
        scopes: Sequence[tuple[bytes | None, bool]],
        *,
        worktree_id: str,
    ) -> _GitFlowSnapshot:
        if self.data_flow is None:
            raise ValidationError("Git data-flow manager is not attached")
        contexts: list[DataFlowContext] = []
        captures: list[tuple[str, str, str]] = []
        selectors: list[tuple[str, str]] = []
        for raw_path, subtree in scopes:
            if raw_path is None:
                if not subtree:
                    continue
                normalized = self._normalized_worktree_root(worktree_id) or "."
            else:
                normalized = self._normalized_file_path(
                    raw_path,
                    worktree_id=worktree_id,
                )
            if subtree:
                context, version = self.data_flow.file_tree_snapshot(normalized)
                kind = "tree"
            else:
                context, version = self.data_flow.file_snapshot(normalized)
                kind = "file"
            contexts.append(context)
            captures.append((kind, normalized, version))
            selectors.append((kind, normalized))

        def resolve() -> str:
            current = [
                (
                    kind,
                    path,
                    self.data_flow.file_tree_state_version(path)
                    if kind == "tree"
                    else self.data_flow.file_state_version(path),
                )
                for kind, path in selectors
            ]
            return self._flow_state_version(current)

        return _GitFlowSnapshot(
            context=DataFlowContext.aggregate(contexts),
            state_version=self._flow_state_version(captures),
            state_version_resolver=resolve,
        )

    def _aggregate_flow_snapshots(
        self,
        snapshots: Sequence[_GitFlowSnapshot],
    ) -> _GitFlowSnapshot:
        selected = tuple(snapshots)
        captures = [
            ("snapshot", str(index), snapshot.state_version)
            for index, snapshot in enumerate(selected)
        ]

        def resolve() -> str:
            return self._flow_state_version(
                [
                    ("snapshot", str(index), snapshot.state_version_resolver())
                    for index, snapshot in enumerate(selected)
                ]
            )

        return _GitFlowSnapshot(
            context=DataFlowContext.aggregate(
                snapshot.context for snapshot in selected
            ),
            state_version=self._flow_state_version(captures),
            state_version_resolver=resolve,
        )

    @staticmethod
    def _canonical_workspace_identity(value: str | os.PathLike[str]) -> bytes:
        resolved = os.fspath(Path(value).resolve(strict=False)).replace("\\", "/")
        canonical = unicodedata.normalize("NFC", resolved).casefold()
        return os.fsencode(canonical)

    def _repository_content_lineage_path(self) -> str:
        workspace = getattr(self.provider, "workspace_root", self.filesystem.root)
        repository = _sha256(self._canonical_workspace_identity(workspace))
        return f".git/agent-libos-flow/{repository}/repository_content/workspace"

    def _repository_content_flow_snapshot(self) -> _GitFlowSnapshot:
        if self.data_flow is None:
            raise ValidationError("Git data-flow manager is not attached")
        path = self._repository_content_lineage_path()
        context, version = self.data_flow.file_snapshot(path)
        return _GitFlowSnapshot(
            context=context,
            state_version=version,
            state_version_resolver=lambda: self.data_flow.file_state_version(path),
        )

    def _bind_repository_content_lineage(
        self,
        *,
        pid: str,
        context: DataFlowContext,
    ) -> None:
        if self.data_flow is None:
            raise ValidationError("Git data-flow manager is not attached")
        self.data_flow.bind_written_file_digest(
            pid=pid,
            normalized_path=self._repository_content_lineage_path(),
            content_sha256=_sha256(
                b"repository_content\0"
                + self._repository_content_lineage_path().encode("utf-8")
            ),
            context=context,
        )

    @staticmethod
    def _git_lineage_path(
        state: GitRepositoryState,
        carrier_kind: str,
        carrier_id: str,
    ) -> str:
        repository = _sha256(state.layout.repository_id.encode("utf-8"))
        carrier = _sha256(
            carrier_kind.encode("ascii")
            + b"\0"
            + carrier_id.encode("ascii")
        )
        return f".git/agent-libos-flow/{repository}/{carrier_kind}/{carrier}"

    @staticmethod
    def _git_lineage_digest(
        state: GitRepositoryState,
        carrier_kind: str,
        carrier_id: str,
    ) -> str:
        return _sha256(
            state.layout.repository_id.encode("utf-8")
            + b"\0"
            + carrier_kind.encode("ascii")
            + b"\0"
            + carrier_id.encode("ascii")
        )

    def _git_lineage_snapshot(
        self,
        state: GitRepositoryState,
        carriers: Sequence[tuple[str, str]],
    ) -> _GitFlowSnapshot:
        if self.data_flow is None:
            raise ValidationError("Git data-flow manager is not attached")
        contexts: list[DataFlowContext] = []
        captures: list[tuple[str, str, str]] = []
        paths: list[str] = []
        for carrier_kind, carrier_id in carriers:
            path = self._git_lineage_path(state, carrier_kind, carrier_id)
            context, version = self.data_flow.file_snapshot(path)
            contexts.append(context)
            captures.append((carrier_kind, path, version))
            paths.append(path)

        def resolve() -> str:
            return self._flow_state_version(
                [
                    (kind, path, self.data_flow.file_state_version(path))
                    for (kind, _carrier_id), path in zip(carriers, paths, strict=True)
                ]
            )

        return _GitFlowSnapshot(
            context=DataFlowContext.aggregate(contexts),
            state_version=self._flow_state_version(captures),
            state_version_resolver=resolve,
        )

    def _current_git_lineage_snapshot(
        self,
        expected_state_token: str,
        *,
        worktree_id: str,
        include_index: bool,
        include_head: bool,
        local_ref: str | None = None,
    ) -> _GitFlowSnapshot:
        state = self.provider.repository_state(
            worktree=self._worktree_path(worktree_id)
        )
        self._require_same_state(expected_state_token, state)
        carriers: list[tuple[str, str]] = []
        if include_index:
            carriers.append(("index", state.index_sha256))
        if local_ref is not None:
            oid = self._resolve_commit(local_ref, worktree_id=worktree_id)
            current = self.provider.repository_state(
                worktree=self._worktree_path(worktree_id)
            )
            self._require_same_state(expected_state_token, current)
            state = current
            carriers.append(("commit", oid))
        elif include_head and state.head_oid is not None:
            carriers.append(("commit", state.head_oid))
        return self._git_lineage_snapshot(state, carriers)

    def _git_read_lineage_snapshot(
        self,
        *,
        worktree_id: str,
        refs: Sequence[str] = (),
        include_head: bool = False,
        include_index: bool = False,
        include_pull_requests: bool = False,
        include_repository_content: bool = False,
        worktree_paths: Sequence[bytes] = (),
        include_worktree_tree: bool = False,
        fixed_carriers: Sequence[tuple[str, str]] = (),
    ) -> _GitReadFlowSnapshot:
        before = self.provider.repository_state(
            worktree=self._worktree_path(worktree_id)
        )
        before_token = self._state_token(before)
        resolved_oids = tuple(
            self._resolve_commit(ref, worktree_id=worktree_id)
            for ref in refs
        )
        state = self.provider.repository_state(
            worktree=self._worktree_path(worktree_id)
        )
        self._require_same_state(before_token.token, state)
        carriers: list[tuple[str, str]] = [*fixed_carriers]
        carriers.extend(("commit", oid) for oid in resolved_oids)
        if include_head and state.head_oid is not None:
            carriers.append(("commit", state.head_oid))
        if include_index:
            carriers.append(("index", state.index_sha256))
        if include_pull_requests:
            carriers.append(("pull_requests", state.pull_requests_sha256))
        selected_carriers = tuple(dict.fromkeys(carriers))
        filesystem_scopes = tuple(
            (path, True) for path in dict.fromkeys(worktree_paths)
        )
        if include_worktree_tree:
            filesystem_scopes = (*filesystem_scopes, (None, True))

        def capture() -> _GitFlowSnapshot:
            snapshots = [self._git_lineage_snapshot(state, selected_carriers)]
            if include_repository_content:
                snapshots.append(self._repository_content_flow_snapshot())
            if filesystem_scopes:
                snapshots.append(
                    self._filesystem_flow_snapshot(
                        filesystem_scopes,
                        worktree_id=worktree_id,
                    )
                )
            return self._aggregate_flow_snapshots(snapshots)

        return _GitReadFlowSnapshot(
            flow=capture(),
            repository_state_token=before_token.token,
            refresh=capture,
        )

    def _diff_read_flow_snapshot(
        self,
        *,
        scope: str,
        base: str | None,
        head: str | None,
        paths: Sequence[bytes],
        worktree_id: str,
        include_patch_index: bool = False,
    ) -> _GitReadFlowSnapshot:
        if scope == "worktree":
            return self._git_read_lineage_snapshot(
                worktree_id=worktree_id,
                include_head=True,
                include_index=True,
                include_repository_content=True,
                worktree_paths=paths,
                include_worktree_tree=not paths,
            )
        if scope == "staged":
            return self._git_read_lineage_snapshot(
                worktree_id=worktree_id,
                include_head=True,
                include_index=True,
                include_repository_content=True,
            )
        if scope == "range" and base is not None and head is not None:
            return self._git_read_lineage_snapshot(
                worktree_id=worktree_id,
                refs=(base, head),
                include_index=include_patch_index,
                include_repository_content=True,
            )
        raise ValidationError("Git diff scope must be worktree, staged, or range")

    def _range_read_flow_snapshot(
        self,
        *,
        base: str,
        head: str,
        worktree_id: str,
        include_index: bool = False,
    ) -> _GitReadFlowSnapshot:
        return self._diff_read_flow_snapshot(
            scope="range",
            base=base,
            head=head,
            paths=(),
            worktree_id=worktree_id,
            include_patch_index=include_index,
        )

    def _log_read_flow_snapshot(
        self,
        *,
        ref: str | None,
        limit: int,
        worktree_id: str,
    ) -> _GitReadFlowSnapshot:
        before = self.provider.repository_state(
            worktree=self._worktree_path(worktree_id)
        )
        before_token = self._state_token(before)
        oid = (
            before.head_oid
            if ref is None
            else self._resolve_commit(ref, worktree_id=worktree_id)
        )
        commit_oids: list[str] = []
        if oid is not None:
            result = self._require_success(
                self._run(
                    [
                        "rev-list",
                        "--parents",
                        f"--max-count={limit + 1}",
                        oid,
                        "--",
                    ],
                    worktree_id=worktree_id,
                    max_output_bytes=self.config.git.output_hard_limit_bytes,
                ),
                "log_flow_snapshot",
            )
            for row in result.stdout.splitlines():
                raw_oids = row.split()
                if not raw_oids:
                    raise GitError(
                        GitErrorCode.COMMAND_FAILED.value,
                        "Git rev-list returned an empty commit record",
                    )
                for raw_oid in raw_oids:
                    selected = raw_oid.decode("ascii", errors="strict")
                    if not _OID_RE.fullmatch(selected):
                        raise GitError(
                            GitErrorCode.COMMAND_FAILED.value,
                            "Git rev-list returned an invalid commit OID",
                        )
                    commit_oids.append(selected)
        current = self.provider.repository_state(
            worktree=self._worktree_path(worktree_id)
        )
        self._require_same_state(before_token.token, current)
        snapshot = self._git_read_lineage_snapshot(
            worktree_id=worktree_id,
            include_repository_content=True,
            fixed_carriers=tuple(
                ("commit", commit_oid) for commit_oid in commit_oids
            ),
        )
        if snapshot.repository_state_token != before_token.token:
            raise GitError(
                GitErrorCode.STALE_STATE.value,
                "Git state changed while log lineage was captured",
                retryable=True,
            )
        return snapshot

    def _show_read_flow_snapshot(
        self,
        *,
        ref: str,
        worktree_id: str,
    ) -> _GitReadFlowSnapshot:
        before = self.provider.repository_state(
            worktree=self._worktree_path(worktree_id)
        )
        before_token = self._state_token(before)
        oid = self._resolve_commit(ref, worktree_id=worktree_id)
        commit = self._commit(oid, worktree_id=worktree_id)
        current = self.provider.repository_state(
            worktree=self._worktree_path(worktree_id)
        )
        self._require_same_state(before_token.token, current)
        snapshot = self._git_read_lineage_snapshot(
            worktree_id=worktree_id,
            include_repository_content=True,
            fixed_carriers=tuple(
                ("commit", commit_oid)
                for commit_oid in dict.fromkeys((oid, *commit.parents))
            ),
        )
        if snapshot.repository_state_token != before_token.token:
            raise GitError(
                GitErrorCode.STALE_STATE.value,
                "Git state changed while show lineage was captured",
                retryable=True,
            )
        return snapshot

    def _blame_read_flow_snapshot(
        self,
        *,
        path: bytes,
        ref: str | None,
        worktree_id: str,
    ) -> _GitReadFlowSnapshot:
        before = self.provider.repository_state(
            worktree=self._worktree_path(worktree_id)
        )
        before_token = self._state_token(before)
        oid = (
            before.head_oid
            if ref is None
            else self._resolve_commit(ref, worktree_id=worktree_id)
        )
        args = ["blame", "--porcelain", "--no-progress"]
        if oid is not None:
            args.append(oid)
        args.extend(["--", os.fsdecode(path)])
        result = self._require_success(
            self._run(
                args,
                worktree_id=worktree_id,
                max_output_bytes=self.config.git.output_hard_limit_bytes,
            ),
            "blame_flow_snapshot",
        )
        commit_oids: list[str] = []
        for line in result.stdout.splitlines():
            if line.startswith(b"previous "):
                fields = line.split(b" ", 2)
                if len(fields) != 3:
                    raise GitError(
                        GitErrorCode.COMMAND_FAILED.value,
                        "Git blame returned an invalid previous record",
                    )
                selected = fields[1].decode("ascii", errors="strict")
                if not _OID_RE.fullmatch(selected):
                    raise GitError(
                        GitErrorCode.COMMAND_FAILED.value,
                        "Git blame returned an invalid previous commit OID",
                    )
                commit_oids.append(selected)
                continue
            fields = line.split(b" ")
            if len(fields) < 3 or not fields[1].isdigit() or not fields[2].isdigit():
                continue
            raw_oid = fields[0].removeprefix(b"^")
            selected = raw_oid.decode("ascii", errors="ignore")
            if _OID_RE.fullmatch(selected) and any(
                character != "0" for character in selected
            ):
                commit_oids.append(selected)
        current = self.provider.repository_state(
            worktree=self._worktree_path(worktree_id)
        )
        self._require_same_state(before_token.token, current)
        snapshot = self._git_read_lineage_snapshot(
            worktree_id=worktree_id,
            include_repository_content=True,
            fixed_carriers=tuple(
                ("commit", commit_oid)
                for commit_oid in dict.fromkeys(commit_oids)
            ),
            worktree_paths=(path,) if oid is None else (),
        )
        if snapshot.repository_state_token != before_token.token:
            raise GitError(
                GitErrorCode.STALE_STATE.value,
                "Git state changed while blame lineage was captured",
                retryable=True,
            )
        return snapshot

    def _current_git_read_flow_snapshot(
        self,
        expected_state_token: str,
        **kwargs: Any,
    ) -> _GitFlowSnapshot:
        snapshot = self._git_read_lineage_snapshot(**kwargs)
        if snapshot.repository_state_token != expected_state_token:
            raise GitError(
                GitErrorCode.STALE_STATE.value,
                "Git state changed before data-flow snapshot",
                retryable=True,
            )
        return snapshot.flow

    def _current_range_flow_snapshot(
        self,
        expected_state_token: str,
        *,
        base: str,
        head: str,
        worktree_id: str,
    ) -> _GitFlowSnapshot:
        snapshot = self._range_read_flow_snapshot(
            base=base,
            head=head,
            worktree_id=worktree_id,
        )
        if snapshot.repository_state_token != expected_state_token:
            raise GitError(
                GitErrorCode.STALE_STATE.value,
                "Git state changed before range data-flow snapshot",
                retryable=True,
            )
        return snapshot.flow

    def _bind_git_lineage(
        self,
        *,
        pid: str,
        state: GitRepositoryState,
        carrier_kind: str,
        carrier_id: str,
        context: DataFlowContext,
        only_if_missing: bool = False,
    ) -> None:
        if self.data_flow is None:
            raise ValidationError("Git data-flow manager is not attached")
        path = self._git_lineage_path(state, carrier_kind, carrier_id)
        if only_if_missing and self.data_flow.store.get_file_label_binding(path) is not None:
            return
        self.data_flow.bind_written_file_digest(
            pid=pid,
            normalized_path=path,
            content_sha256=self._git_lineage_digest(
                state,
                carrier_kind,
                carrier_id,
            ),
            context=context,
        )

    def _filesystem_authorizations(
        self,
        *,
        pid: str,
        operation: str,
        worktree_id: str,
        paths: Sequence[bytes],
        rights: Sequence[CapabilityRight],
        base_context: dict[str, Any],
        source_oids: Iterable[str] | None,
        path_subtree: bool = False,
    ) -> tuple[list[CapabilityDecision], tuple[tuple[bytes | None, bool], ...]]:
        decisions: list[CapabilityDecision] = []
        if not rights:
            return decisions, ()
        targets: tuple[bytes | None, ...] = tuple(paths) if paths else (None,)
        scopes = tuple(
            (
                path,
                path is None
                or path_subtree
                or self.provider.preflight_path_kind(
                    path,
                    worktree=self._worktree_path(worktree_id),
                )
                in {"directory", "missing"},
            )
            for path in targets
        )
        for right in rights:
            for path, subtree in scopes:
                resource = self._filesystem_resource(
                    path,
                    worktree_id=worktree_id,
                    subtree=subtree,
                )
                path_sha256 = _sha256(path) if path is not None else None
                context = {
                    **base_context,
                    "primitive": f"runtime.git.{operation}",
                    "authority_operation": f"filesystem.{right.value}",
                    "resource": resource,
                    "right": right.value,
                    "path_sha256": path_sha256,
                    "path_scope": "subtree" if subtree else "exact",
                }
                decisions.append(
                    self._authorize(
                        pid=pid,
                        resource=resource,
                        right=right,
                        context=context,
                        question=f"Allow Git {operation.replace('_', ' ')} to affect an authorized workspace path?",
                        source_oids=source_oids,
                    )
                )
        return decisions, scopes

    def _mutation_command(
        self,
        before: GitRepositoryState,
        operation: str,
        args: Sequence[str],
        *,
        worktree_id: str,
        stdin: bytes | None = None,
        remote: str | None = None,
        expected_remote_fingerprint: str | None = None,
        verify_after: bool = True,
    ) -> GitCommandResult:
        current_before = self.provider.repository_state(worktree=self._worktree_path(worktree_id))
        if self._state_token(current_before).token != self._state_token(before).token:
            raise GitError(
                GitErrorCode.STALE_STATE.value,
                "Git repository state changed immediately before mutation dispatch",
                operation=operation,
                retryable=True,
            )
        self.provider.validate_operation(
            str(args[0]),
            worktree=self._worktree_path(worktree_id),
            remote=remote,
        )
        self._mark_git_effect_started()
        result = self._run(
            args,
            worktree_id=worktree_id,
            max_output_bytes=self.config.git.output_hard_limit_bytes,
            remote=remote,
            expected_remote_fingerprint=expected_remote_fingerprint,
            read_only=False,
            stdin=stdin,
            verify_after=verify_after,
        )
        if result.returncode != 0:
            current = self.provider.repository_state(worktree=self._worktree_path(worktree_id))
            effect = (
                "none"
                if self._state_token(before).token == self._state_token(current).token
                else "unknown"
            )
            raise self._error_from_result(result, operation, effect=effect)
        return result

    @staticmethod
    def _provider_not_started(exc: GitError) -> GitProviderEffectNotStarted:
        return GitProviderEffectNotStarted(
            exc.code,
            str(exc),
            operation=exc.operation,
            retryable=exc.retryable,
            details=exc.details,
        )

    def _mark_git_effect_started(self) -> None:
        state = getattr(self._dispatch_state, "current", None)
        if isinstance(state, dict):
            if not state.get("started"):
                self._prebind_repository_content_lineage()
            state["started"] = True

    def _mark_git_materialized_worktree(self, worktree_id: str) -> None:
        if not _WORKTREE_ID_RE.fullmatch(worktree_id) or worktree_id == "main":
            raise ValidationError("invalid materialized Git worktree id")
        state = getattr(self._dispatch_state, "current", None)
        if isinstance(state, dict):
            state["materialized_worktree_id"] = worktree_id

    def _mark_git_source_carrier(self, carrier_kind: str, carrier_id: str) -> None:
        if carrier_kind == "commit":
            valid = bool(_OID_RE.fullmatch(carrier_id))
        elif carrier_kind in {"index", "pull_requests"}:
            valid = bool(_SHA256_RE.fullmatch(carrier_id))
        elif carrier_kind == "pull_request":
            valid = bool(_PR_ID_RE.fullmatch(carrier_id))
        else:
            valid = False
        if not valid:
            raise ValidationError("invalid Git data-flow source carrier")
        state = getattr(self._dispatch_state, "current", None)
        if isinstance(state, dict):
            carriers = state.setdefault("source_carriers", [])
            if isinstance(carriers, list):
                carriers.append((carrier_kind, carrier_id))
                if state.get("started"):
                    self._prebind_repository_content_lineage()

    def _mark_git_output_carrier(self, carrier_kind: str, carrier_id: str) -> None:
        if carrier_kind != "pull_request" or not _PR_ID_RE.fullmatch(carrier_id):
            raise ValidationError("invalid Git data-flow output carrier")
        state = getattr(self._dispatch_state, "current", None)
        if isinstance(state, dict):
            carriers = state.setdefault("output_carriers", [])
            if isinstance(carriers, list):
                carriers.append((carrier_kind, carrier_id))

    def _prebind_pull_request_lineage(self, pr_id: str) -> None:
        state = getattr(self._dispatch_state, "current", None)
        if not isinstance(state, dict):
            raise ValidationError(
                "pull request lineage must be bound inside Git mutation dispatch"
            )
        pid = state.get("pid")
        repository_state = state.get("repository_state")
        mutation_context = state.get("mutation_context")
        if (
            not isinstance(pid, str)
            or not isinstance(repository_state, GitRepositoryState)
            or not isinstance(mutation_context, DataFlowContext)
        ):
            raise ValidationError("Git mutation data-flow state is incomplete")
        contexts = [
            mutation_context,
            *tuple(state.get("source_contexts") or ()),
        ]
        source_carriers = tuple(
            dict.fromkeys(tuple(state.get("source_carriers") or ()))
        )
        if source_carriers:
            contexts.append(
                self._git_lineage_snapshot(
                    repository_state,
                    source_carriers,
                ).context
            )
        selected_context = DataFlowContext.aggregate(contexts)
        if self.data_flow is None:
            raise ValidationError("Git data-flow manager is not attached")
        with self.data_flow.store.transaction():
            for carrier_kind, carrier_id in (
                ("pull_request", pr_id),
                ("pull_request_collection", "workspace"),
            ):
                self._bind_git_lineage(
                    pid=pid,
                    state=repository_state,
                    carrier_kind=carrier_kind,
                    carrier_id=carrier_id,
                    context=selected_context,
                )

    def _prebind_repository_content_lineage(self) -> None:
        state = getattr(self._dispatch_state, "current", None)
        if not isinstance(state, dict):
            raise ValidationError(
                "repository content lineage must be bound inside Git mutation dispatch"
            )
        pid = state.get("pid")
        repository_state = state.get("repository_state")
        mutation_context = state.get("mutation_context")
        if (
            not isinstance(pid, str)
            or not isinstance(repository_state, GitRepositoryState)
            or not isinstance(mutation_context, DataFlowContext)
        ):
            raise ValidationError("Git mutation data-flow state is incomplete")
        contexts = [mutation_context, *tuple(state.get("source_contexts") or ())]
        source_carriers = tuple(
            dict.fromkeys(tuple(state.get("source_carriers") or ()))
        )
        if source_carriers:
            contexts.append(
                self._git_lineage_snapshot(
                    repository_state,
                    source_carriers,
                ).context
            )
        self._bind_repository_content_lineage(
            pid=pid,
            context=DataFlowContext.aggregate(contexts),
        )

    def _mark_git_source_context(self, context: DataFlowContext) -> None:
        state = getattr(self._dispatch_state, "current", None)
        if isinstance(state, dict):
            contexts = state.setdefault("source_contexts", [])
            if isinstance(contexts, list):
                contexts.append(context)
                if state.get("started"):
                    self._prebind_repository_content_lineage()

    def _resolve_stash_commit(
        self,
        index: int,
        *,
        worktree_id: str,
        optional: bool = False,
    ) -> str | None:
        if isinstance(index, bool) or index < 0 or index > 100_000:
            raise ValidationError("Git stash index is invalid")
        selected = f"stash@{{{index}}}"
        result = self._run(
            [
                "rev-parse",
                "--verify",
                "--end-of-options",
                f"{selected}^{{commit}}",
            ],
            worktree_id=worktree_id,
            max_output_bytes=65536,
        )
        if optional and result.returncode != 0:
            return None
        self._require_success(result, "resolve_stash")
        oid = result.stdout.strip().decode("ascii", errors="strict")
        if not _OID_RE.fullmatch(oid):
            raise GitError(
                GitErrorCode.INVALID_REF.value,
                "Git returned an invalid stash object id",
            )
        return oid

    def _revalidate_mutation_authority(
        self,
        decisions: Sequence[CapabilityDecision],
        mandatory_decision_indexes: set[int],
    ) -> tuple[CapabilityDecision, ...]:
        current: list[CapabilityDecision] = []
        for index, original in enumerate(decisions):
            decision = self.capabilities.reauthorize_decision(original)
            if (
                index in mandatory_decision_indexes
                and decision.allowed
                and not self._selected_capability_has_approval_binding(decision)
            ):
                raise CapabilityDenied(
                    "mandatory Git approval binding changed before dispatch"
                )
            current.append(decision)
        return tuple(current)

    def _authorize_mutation(
        self,
        *,
        pid: str,
        operation: str,
        target: str,
        primary_right: CapabilityRight,
        context: dict[str, Any],
        destructive: bool,
        additional_authorities: Sequence[tuple[str, CapabilityRight, bool]],
        mandatory_primary_approval: bool,
        source_oids: Iterable[str] | None,
        worktree_id: str,
        paths: Sequence[bytes],
        filesystem_rights: Sequence[CapabilityRight],
        filesystem_path_subtree: bool,
    ) -> tuple[
        list[CapabilityDecision],
        set[int],
        dict[str, Any] | None,
        tuple[tuple[bytes | None, bool], ...],
    ]:
        decisions = [
            self._authorize(
                pid=pid,
                resource=target,
                right=primary_right,
                context=context,
                question=f"Allow this process to perform Git {operation.replace('_', ' ')}?",
                mandatory_approval=mandatory_primary_approval,
                source_oids=source_oids,
            )
        ]
        mandatory_indexes = {0} if mandatory_primary_approval else set()
        approval_context = context if mandatory_primary_approval else None
        if destructive:
            for right in (CapabilityRight.DELETE, CapabilityRight.ADMIN):
                destructive_context = {
                    **context,
                    "right": right.value,
                    "authority_operation": f"git.{right.value}",
                }
                decisions.append(
                    self._authorize(
                        pid=pid,
                        resource=target,
                        right=right,
                        context=destructive_context,
                        question=f"Approve destructive Git {operation.replace('_', ' ')} for this exact state?",
                        mandatory_approval=right is CapabilityRight.ADMIN,
                        source_oids=source_oids,
                    )
                )
                if right is CapabilityRight.ADMIN:
                    mandatory_indexes.add(len(decisions) - 1)
                    approval_context = destructive_context
        for additional_resource, additional_right, mandatory in additional_authorities:
            additional_context = {
                **context,
                "resource": additional_resource,
                "right": additional_right.value,
                "authority_operation": f"git.{additional_right.value}",
            }
            decisions.append(
                self._authorize(
                    pid=pid,
                    resource=additional_resource,
                    right=additional_right,
                    context=additional_context,
                    question=f"Allow Git {operation.replace('_', ' ')} on {additional_resource}?",
                    mandatory_approval=mandatory,
                    source_oids=source_oids,
                )
            )
            if mandatory:
                if approval_context is not None:
                    raise ValidationError(
                        "a Git operation must use one aggregate human approval binding"
                    )
                mandatory_indexes.add(len(decisions) - 1)
                approval_context = additional_context
        filesystem_decisions, filesystem_scopes = self._filesystem_authorizations(
            pid=pid,
            operation=operation,
            worktree_id=worktree_id,
            paths=paths,
            rights=filesystem_rights,
            base_context=context,
            source_oids=source_oids,
            path_subtree=filesystem_path_subtree,
        )
        decisions.extend(filesystem_decisions)
        return decisions, mandatory_indexes, approval_context, filesystem_scopes

    def _mutation_context(
        self,
        *,
        pid: str,
        operation: str,
        expected_state_token: str,
        resource: str | None,
        primary_right: CapabilityRight,
        worktree_id: str,
        paths: Sequence[bytes],
        extra: dict[str, Any] | None,
    ) -> tuple[str, str, dict[str, Any]]:
        expected = self._validate_expected_token(expected_state_token)
        target = resource or self.repository_resource
        context = self._operation_context(
            pid=pid,
            operation=operation,
            resource=target,
            right=primary_right,
            worktree_id=worktree_id,
            extra={
                "target_state_version": expected,
                "expected_state_token": expected,
                "git_expected_state_token": expected,
                "path_count": len(paths),
                "paths_sha256": _sha256(b"\0".join(paths)),
                **dict(extra or {}),
            },
        )
        return expected, target, context

    def _mutation_invocation(
        self,
        *,
        pid: str,
        operation: str,
        target: str,
        selected_descriptor: str,
        expected: str,
        context: dict[str, Any],
        decisions: list[CapabilityDecision],
        mandatory_indexes: set[int],
        approval_context: dict[str, Any] | None,
        observation: dict[str, Any],
        source_oids: Iterable[str] | None,
        flow_snapshot: _GitFlowSnapshot | None,
        additional_data_sinks: Sequence[DataSink],
    ) -> tuple[ProtectedOperationInvocation, DataFlowContext, DataFlowContext | None]:
        if self.data_flow is None:
            raise ValidationError("Git data-flow manager is not attached")
        flow_context = DataFlowContext.aggregate(
            (
                self.data_flow.context_from_source_oids(pid, source_oids),
                *(() if flow_snapshot is None else (flow_snapshot.context,)),
            )
        )
        ingress_context = (
            None
            if selected_descriptor == "primitive.git.push"
            else self.data_flow.unclassified_ingress_context(
                flow_context,
                origin="external:git",
            )
        )
        target_state_version = (
            expected
            if flow_snapshot is None
            else self._combined_flow_state_version(
                expected,
                flow_snapshot.state_version,
            )
        )
        target_state_resolver = (
            None
            if flow_snapshot is None
            else lambda: self._combined_flow_state_version(
                expected,
                flow_snapshot.state_version_resolver(),
            )
        )
        invocation = ProtectedOperationInvocation(
            pid=pid,
            actor=pid,
            target=target,
            decisions=tuple(decisions),
            canonical_args=approval_context or context,
            observation=observation,
            authority_revalidator=lambda: self._revalidate_mutation_authority(
                decisions,
                mandatory_indexes,
            ),
            data_sink=DataSink(target),
            additional_data_sinks=tuple(additional_data_sinks),
            data_flow_context=flow_context,
            data_flow_ingress_context=ingress_context,
            data_flow_payload=context,
            data_flow_operation=f"git.{operation}",
            data_flow_target_state_version=target_state_version,
            data_flow_target_state_version_resolver=target_state_resolver,
        )
        return invocation, flow_context, ingress_context

    def _protected_flow_snapshot(
        self,
        *,
        pid: str,
        operation: str,
        worktree_id: str,
        scope_count: int,
        resolver: Callable[[], _GitFlowSnapshot],
    ) -> _GitFlowSnapshot:
        def read_snapshot() -> tuple[_GitFlowSnapshot, dict[str, Any]]:
            snapshot = resolver()
            return snapshot, {
                "scope_count": scope_count,
                "state_version": snapshot.state_version,
            }

        return self._read(
            pid,
            f"{operation}_flow_snapshot",
            read_snapshot,
            worktree_id=worktree_id,
        )

    def _complete_mutation(
        self,
        *,
        pid: str,
        operation: str,
        descriptor: str,
        target: str,
        remote: str | None,
        invocation: ProtectedOperationInvocation,
        observation: dict[str, Any],
        input_refs: Iterable[str],
        worktree_id: str,
        flow_lock_paths: Sequence[str],
        dispatch: Callable[[], tuple[GitOperationResult, dict[str, Any]]],
    ) -> GitOperationResult:
        def provider_phase() -> tuple[GitOperationResult, dict[str, Any]]:
            return dispatch()

        with self.protected_operations.start(
            descriptor,
            invocation,
            provider=self.provider,
        ) as protected:
            with self.provider.repository_lock(
                worktree=self._worktree_path(worktree_id)
            ):
                with self.filesystem.hold_file_label_io_paths(flow_lock_paths):
                    result, summary = protected.call(
                        ProviderPhase(
                            operation,
                            state_mutation=True,
                            information_flow=remote is not None,
                        ),
                        provider_phase,
                    )
                    return protected.complete(
                        result,
                        self._evidence(
                            pid=pid,
                            operation=operation,
                            resource=target,
                            summary=summary,
                            mutation=True,
                            input_refs=input_refs,
                        ),
                        classification_context=observation,
                        classification_result=summary,
                    )

    def _status_worktree_paths(self, state: GitRepositoryState) -> tuple[bytes, ...]:
        parsed = self._parse_status(
            state,
            limit=self.config.git.status_entry_hard_limit,
        )
        paths: list[bytes] = []
        for entry in parsed.entries:
            paths.append(self._decode_path(entry.path))
            if entry.original_path is not None:
                paths.append(self._decode_path(entry.original_path))
        return tuple(dict.fromkeys(paths))

    def _changed_worktree_paths(
        self,
        before: GitRepositoryState,
        after: GitRepositoryState,
        *,
        worktree_id: str,
    ) -> tuple[bytes, ...]:
        paths = [
            *self._status_worktree_paths(before),
            *self._status_worktree_paths(after),
        ]
        if (
            before.head_oid is not None
            and after.head_oid is not None
            and before.head_oid != after.head_oid
        ):
            changed = self._require_success(
                self._run(
                    [
                        "diff",
                        "--name-only",
                        "-z",
                        "--no-renames",
                        before.head_oid,
                        after.head_oid,
                        "--",
                    ],
                    worktree_id=worktree_id,
                    max_output_bytes=self.config.git.output_hard_limit_bytes,
                ),
                "changed_worktree_paths",
            )
            paths.extend(
                self._decode_path(path)
                for path in self._bounded_git_paths(changed.stdout)
            )
        return tuple(dict.fromkeys(paths))

    def _tracked_worktree_paths(self, worktree_id: str) -> tuple[bytes, ...]:
        listed = self._require_success(
            self._run(
                ["ls-files", "-z"],
                worktree_id=worktree_id,
                max_output_bytes=self.config.git.output_hard_limit_bytes,
            ),
            "list_tracked_worktree_paths",
        )
        return tuple(
            self._decode_path(path)
            for path in self._bounded_git_paths(listed.stdout)
        )

    def _reconcile_worktree_labels(
        self,
        *,
        pid: str,
        worktree_id: str,
        paths: Sequence[bytes],
        context: DataFlowContext,
    ) -> None:
        if self.data_flow is None:
            raise ValidationError("Git data-flow manager is not attached")
        with self.data_flow.store.transaction():
            for raw_path in paths:
                normalized = self._normalized_file_path(
                    raw_path,
                    worktree_id=worktree_id,
                )
                kind = self.provider.path_kind(
                    raw_path,
                    worktree=self._worktree_path(worktree_id),
                )
                if kind == "missing":
                    bindings = self.data_flow.store.list_file_label_bindings_for_tree(
                        normalized
                    )
                    self.data_flow.tombstone_path_tree(
                        pid=pid,
                        expected_bindings={
                            item.normalized_path: (item.binding_id, item.generation)
                            for item in bindings
                        },
                    )
                    continue
                if kind == "directory":
                    binding = self.data_flow.store.get_file_label_binding(normalized)
                    if binding is not None:
                        self.data_flow.tombstone_file(
                            pid=pid,
                            normalized_path=normalized,
                            expected_binding_id=binding.binding_id,
                            expected_generation=binding.generation,
                        )
                    continue
                digest = self.provider.path_content_sha256(
                    raw_path,
                    worktree=self._worktree_path(worktree_id),
                )
                if digest is None:
                    binding = self.data_flow.store.get_file_label_binding(normalized)
                    if binding is not None:
                        self.data_flow.tombstone_file(
                            pid=pid,
                            normalized_path=normalized,
                            expected_binding_id=binding.binding_id,
                            expected_generation=binding.generation,
                        )
                    continue
                binding = self.data_flow.store.get_file_label_binding(normalized)
                if binding is not None and binding.content_sha256 == digest:
                    continue
                self.data_flow.bind_written_file_digest(
                    pid=pid,
                    normalized_path=normalized,
                    content_sha256=digest,
                    context=context,
                )

    def _settle_git_lineage(
        self,
        *,
        pid: str,
        before: GitRepositoryState,
        after: GitRepositoryState,
        created_oid: str | None,
        context: DataFlowContext,
    ) -> None:
        if self.data_flow is None:
            raise ValidationError("Git data-flow manager is not attached")
        with self.data_flow.store.transaction():
            self._bind_repository_content_lineage(
                pid=pid,
                context=context,
            )
            if before.index_sha256 != after.index_sha256:
                previous_index = self._git_lineage_snapshot(
                    before,
                    (("index", before.index_sha256),),
                )
                self._bind_git_lineage(
                    pid=pid,
                    state=after,
                    carrier_kind="index",
                    carrier_id=after.index_sha256,
                    context=DataFlowContext.aggregate(
                        (previous_index.context, context)
                    ),
                )
            if before.pull_requests_sha256 != after.pull_requests_sha256:
                previous_pull_requests = self._git_lineage_snapshot(
                    before,
                    (("pull_requests", before.pull_requests_sha256),),
                )
                self._bind_git_lineage(
                    pid=pid,
                    state=after,
                    carrier_kind="pull_requests",
                    carrier_id=after.pull_requests_sha256,
                    context=DataFlowContext.aggregate(
                        (previous_pull_requests.context, context)
                    ),
                )
            if created_oid is None or not _OID_RE.fullmatch(created_oid):
                return
            carriers: list[tuple[str, str]] = [
                ("index", after.index_sha256),
            ]
            if before.head_oid is not None:
                carriers.append(("commit", before.head_oid))
            sources = self._git_lineage_snapshot(after, carriers)
            self._bind_git_lineage(
                pid=pid,
                state=after,
                carrier_kind="commit",
                carrier_id=created_oid,
                context=DataFlowContext.aggregate((sources.context, context)),
                only_if_missing=True,
            )

    def _settle_mutation_state(
        self,
        *,
        pid: str,
        before: GitRepositoryState,
        after: GitRepositoryState,
        created_oid: str | None,
        changed_raw: Sequence[bytes],
        reconciliation_paths: Sequence[bytes],
        writes_worktree: bool,
        worktree_id: str,
        context: DataFlowContext,
        source_carriers: Sequence[tuple[str, str]],
        source_contexts: Sequence[DataFlowContext],
        output_carriers: Sequence[tuple[str, str]],
        materialized_worktree_id: str | None,
    ) -> tuple[bytes, ...]:
        selected_carriers = tuple(dict.fromkeys(source_carriers))
        contexts = [context, *source_contexts]
        if selected_carriers:
            sources = self._git_lineage_snapshot(after, selected_carriers)
            contexts.append(sources.context)
        settled_context = DataFlowContext.aggregate(contexts)
        changed = tuple(
            dict.fromkeys(
                (
                    *changed_raw,
                    *reconciliation_paths,
                    *(
                        self._changed_worktree_paths(
                            before,
                            after,
                            worktree_id=worktree_id,
                        )
                        if writes_worktree
                        else ()
                    ),
                )
            )
        )
        self._settle_git_lineage(
            pid=pid,
            before=before,
            after=after,
            created_oid=created_oid,
            context=settled_context,
        )
        for carrier_kind, carrier_id in tuple(dict.fromkeys(output_carriers)):
            self._bind_git_lineage(
                pid=pid,
                state=after,
                carrier_kind=carrier_kind,
                carrier_id=carrier_id,
                context=settled_context,
            )
        if writes_worktree:
            self._reconcile_worktree_labels(
                pid=pid,
                worktree_id=worktree_id,
                paths=changed,
                context=settled_context,
            )
        if materialized_worktree_id is not None:
            self.provider.repository_layout(
                worktree=self._worktree_path(materialized_worktree_id)
            )
            self._reconcile_worktree_labels(
                pid=pid,
                worktree_id=materialized_worktree_id,
                paths=self._tracked_worktree_paths(materialized_worktree_id),
                context=settled_context,
            )
        return changed

    def _revalidate_exact_path_scopes(
        self,
        scopes: Sequence[tuple[bytes | None, bool]],
        *,
        worktree_id: str,
    ) -> None:
        for raw_path, subtree in scopes:
            if raw_path is None or subtree:
                continue
            if self.provider.path_kind(
                raw_path,
                worktree=self._worktree_path(worktree_id),
            ) == "directory":
                raise GitError(
                    GitErrorCode.STALE_STATE.value,
                    "Git path changed from an exact file to a directory before dispatch",
                    retryable=True,
                )

    def _settle_failed_mutation(
        self,
        *,
        pid: str,
        before: GitRepositoryState,
        worktree_id: str,
        reconciliation_paths: Sequence[bytes],
        writes_worktree: bool,
        mutation_context: DataFlowContext,
        current_state: dict[str, Any],
        materialized_worktree_id: str | None,
        operation_error: BaseException,
    ) -> None:
        try:
            after = self.provider.repository_state(
                worktree=self._worktree_path(worktree_id)
            )
            self._settle_mutation_state(
                pid=pid,
                before=before,
                after=after,
                created_oid=(
                    after.head_oid if after.head_oid != before.head_oid else None
                ),
                changed_raw=(),
                reconciliation_paths=reconciliation_paths,
                writes_worktree=writes_worktree,
                worktree_id=worktree_id,
                context=mutation_context,
                source_carriers=tuple(current_state["source_carriers"]),
                source_contexts=tuple(current_state["source_contexts"]),
                output_carriers=tuple(current_state["output_carriers"]),
                materialized_worktree_id=(
                    materialized_worktree_id
                    if current_state.get("materialized_worktree_id")
                    == materialized_worktree_id
                    else None
                ),
            )
        except BaseException as settlement_error:
            raise BaseExceptionGroup(
                "Git mutation failed and its data-flow state could not be reconciled",
                [operation_error, settlement_error],
            ) from None

    def _dispatch_mutation(
        self,
        *,
        pid: str,
        operation: str,
        expected: str,
        worktree_id: str,
        paths: Sequence[bytes],
        reconciliation_paths: Sequence[bytes],
        materialized_worktree_id: str | None,
        filesystem_rights: Sequence[CapabilityRight],
        filesystem_path_scopes: Sequence[tuple[bytes | None, bool]],
        mutation_context: DataFlowContext,
        callback: _GitMutationCallback,
    ) -> tuple[GitOperationResult, dict[str, Any]]:
        previous_state = getattr(self._dispatch_state, "current", None)
        current_state: dict[str, Any] = {
            "started": False,
            "source_carriers": [],
            "source_contexts": [],
            "output_carriers": [],
            "materialized_worktree_id": None,
        }
        self._dispatch_state.current = current_state
        try:
            # _complete_mutation holds the repository and filesystem label
            # locks through provider dispatch, settlement, and completion.
            before_state = self.provider.repository_state(
                worktree=self._worktree_path(worktree_id)
            )
            current_state.update(
                pid=pid,
                repository_state=before_state,
                mutation_context=mutation_context,
            )
            before_token = self._require_same_state(expected, before_state)
            self._revalidate_exact_path_scopes(
                filesystem_path_scopes,
                worktree_id=worktree_id,
            )
            writes_worktree = any(
                right in {CapabilityRight.WRITE, CapabilityRight.DELETE}
                for right in filesystem_rights
            )
            try:
                created_oid, details, changed_raw = callback(before_state)
            except BaseException as operation_error:
                if not current_state["started"]:
                    if (
                        isinstance(operation_error, GitError)
                        and operation_error.code == GitErrorCode.STALE_STATE.value
                        and operation_error.retryable
                    ):
                        operation_error.details.setdefault(
                            "effect_started",
                            False,
                        )
                    raise
                self._settle_failed_mutation(
                    pid=pid,
                    before=before_state,
                    worktree_id=worktree_id,
                    reconciliation_paths=reconciliation_paths,
                    writes_worktree=writes_worktree,
                    mutation_context=mutation_context,
                    current_state=current_state,
                    materialized_worktree_id=materialized_worktree_id,
                    operation_error=operation_error,
                )
                raise
            after_state = self.provider.repository_state(
                worktree=self._worktree_path(worktree_id)
            )
            changed = self._settle_mutation_state(
                pid=pid,
                before=before_state,
                after=after_state,
                created_oid=created_oid,
                changed_raw=changed_raw,
                reconciliation_paths=reconciliation_paths,
                writes_worktree=writes_worktree,
                worktree_id=worktree_id,
                context=mutation_context,
                source_carriers=tuple(current_state["source_carriers"]),
                source_contexts=tuple(current_state["source_contexts"]),
                output_carriers=tuple(current_state["output_carriers"]),
                materialized_worktree_id=(
                    materialized_worktree_id
                    if current_state.get("materialized_worktree_id")
                    == materialized_worktree_id
                    else None
                ),
            )
            after_token = self._state_token(after_state)
            result = GitOperationResult(
                operation=operation,
                repository_id=before_state.layout.repository_id,
                worktree_id=before_state.layout.worktree_id,
                before=before_token,
                after=after_token,
                changed_paths=[self._git_path(path) for path in changed],
                created_oid=created_oid,
                details=dict(details),
            )
            return result, {
                "repository_id": result.repository_id,
                "worktree_id": result.worktree_id,
                "before_state_token": before_token.token,
                "after_state_token": after_token.token,
                "changed_paths": len(result.changed_paths),
                "changed_paths_sha256": _sha256(b"\0".join(changed)),
                "created_oid": created_oid,
                **dict(details),
            }
        except GitProviderEffectNotStarted:
            raise
        except GitError as exc:
            if not current_state["started"]:
                raise self._provider_not_started(exc) from exc
            raise
        finally:
            if previous_state is None:
                with contextlib.suppress(AttributeError):
                    del self._dispatch_state.current
            else:
                self._dispatch_state.current = previous_state

    @_with_provider_subprocess_scope
    def _mutate(
        self,
        pid: str,
        operation: str,
        expected_state_token: str,
        callback: _GitMutationCallback,
        *,
        worktree_id: str = "main",
        paths: Sequence[bytes] = (),
        reconciliation_paths: Sequence[bytes] = (),
        materialized_worktree_id: str | None = None,
        filesystem_rights: Sequence[CapabilityRight] = (),
        destructive: bool = False,
        extra: dict[str, Any] | None = None,
        descriptor: str = "primitive.git.mutate",
        resource: str | None = None,
        primary_right: CapabilityRight = CapabilityRight.WRITE,
        remote: str | None = None,
        source_oids: Iterable[str] | None = None,
        input_refs: Iterable[str] = (),
        filesystem_path_subtree: bool = False,
        additional_authorities: Sequence[tuple[str, CapabilityRight, bool]] = (),
        mandatory_primary_approval: bool = False,
        flow_snapshot_resolver: _GitFlowSnapshotResolver | None = None,
        protected_flow_snapshot_resolver: _GitFlowSnapshotResolver | None = None,
        additional_data_sinks: Sequence[DataSink] = (),
    ) -> GitOperationResult:
        if descriptor not in _GIT_MUTATION_DESCRIPTORS:
            raise ValidationError("unsupported protected Git operation descriptor")
        if (
            flow_snapshot_resolver is not None
            and protected_flow_snapshot_resolver is not None
        ):
            raise ValidationError("Git mutation accepts only one flow snapshot resolver")
        expected, target, context = self._mutation_context(
            pid=pid,
            operation=operation,
            expected_state_token=expected_state_token,
            resource=resource,
            primary_right=primary_right,
            worktree_id=worktree_id,
            paths=paths,
            extra=extra,
        )
        decisions, mandatory_indexes, approval_context, filesystem_path_scopes = self._authorize_mutation(
            pid=pid,
            operation=operation,
            target=target,
            primary_right=primary_right,
            context=context,
            destructive=destructive,
            additional_authorities=additional_authorities,
            mandatory_primary_approval=mandatory_primary_approval,
            source_oids=source_oids,
            worktree_id=worktree_id,
            paths=paths,
            filesystem_rights=filesystem_rights,
            filesystem_path_subtree=filesystem_path_subtree,
        )
        observation = {
            "read_only": False,
            "remote": remote,
            "worktree_id": worktree_id,
            "expected_state_token": expected,
            **dict(extra or {}),
        }
        flow_snapshot = None
        if flow_snapshot_resolver is not None:
            flow_snapshot = flow_snapshot_resolver(filesystem_path_scopes)
        elif protected_flow_snapshot_resolver is not None:
            def resolve_protected_snapshot() -> _GitFlowSnapshot:
                return protected_flow_snapshot_resolver(filesystem_path_scopes)

            flow_snapshot = self._protected_flow_snapshot(
                pid=pid,
                operation=operation,
                worktree_id=worktree_id,
                scope_count=len(filesystem_path_scopes),
                resolver=resolve_protected_snapshot,
            )
        flow_snapshot = self._aggregate_flow_snapshots(
            (
                self._repository_content_flow_snapshot(),
                *(() if flow_snapshot is None else (flow_snapshot,)),
            )
        )
        invocation, flow_context, ingress_context = self._mutation_invocation(
            pid=pid,
            operation=operation,
            target=target,
            selected_descriptor=descriptor,
            expected=expected,
            context=context,
            decisions=decisions,
            mandatory_indexes=mandatory_indexes,
            approval_context=approval_context,
            observation=observation,
            source_oids=source_oids,
            flow_snapshot=flow_snapshot,
            additional_data_sinks=additional_data_sinks,
        )

        def dispatch() -> tuple[GitOperationResult, dict[str, Any]]:
            return self._dispatch_mutation(
                operation=operation,
                expected=expected,
                worktree_id=worktree_id,
                paths=paths,
                reconciliation_paths=reconciliation_paths,
                materialized_worktree_id=materialized_worktree_id,
                filesystem_rights=filesystem_rights,
                filesystem_path_scopes=filesystem_path_scopes,
                pid=pid,
                mutation_context=ingress_context or flow_context,
                callback=lambda before: callback(before),
            )

        writes_worktree = any(
            right in {CapabilityRight.WRITE, CapabilityRight.DELETE}
            for right in filesystem_rights
        )
        if writes_worktree:
            flow_lock_paths = (self._normalized_worktree_root(worktree_id) or ".",)
        elif filesystem_rights:
            lock_paths = tuple(paths) if paths else (None,)
            flow_lock_paths = tuple(
                self._normalized_worktree_root(worktree_id) or "."
                if raw_path is None
                else self._normalized_file_path(
                    raw_path,
                    worktree_id=worktree_id,
                )
                for raw_path in lock_paths
            )
        else:
            flow_lock_paths = ()

        return self._complete_mutation(
            pid=pid,
            operation=operation,
            descriptor=descriptor,
            target=target,
            remote=remote,
            invocation=invocation,
            observation=observation,
            input_refs=input_refs,
            worktree_id=worktree_id,
            flow_lock_paths=flow_lock_paths,
            dispatch=dispatch,
        )

    def stage(
        self,
        pid: str,
        paths: Iterable[str | GitPath | dict[str, Any]],
        expected_state_token: str,
        *,
        worktree_id: str = "main",
    ) -> GitOperationResult:
        selected = self._decode_paths(paths, required=True)

        def dispatch(before: GitRepositoryState) -> tuple[str | None, dict[str, Any], Sequence[bytes]]:
            self._mutation_command(
                before,
                "stage",
                ["add", "--", *self._path_argv(selected)],
                worktree_id=worktree_id,
            )
            return None, {"staged_paths": len(selected)}, selected

        return self._mutate(
            pid,
            "stage",
            expected_state_token,
            dispatch,
            worktree_id=worktree_id,
            paths=selected,
            filesystem_rights=(CapabilityRight.READ,),
            flow_snapshot_resolver=lambda scopes: self._filesystem_flow_snapshot(
                scopes,
                worktree_id=worktree_id,
            ),
        )

    def unstage(
        self,
        pid: str,
        paths: Iterable[str | GitPath | dict[str, Any]],
        expected_state_token: str,
        *,
        worktree_id: str = "main",
    ) -> GitOperationResult:
        selected = self._decode_paths(paths, required=True)

        def dispatch(before: GitRepositoryState) -> tuple[str | None, dict[str, Any], Sequence[bytes]]:
            if before.head_oid is None:
                args = ["rm", "--cached", "-r", "-f", "--ignore-unmatch", "--", *self._path_argv(selected)]
            else:
                args = ["reset", "-q", before.head_oid, "--", *self._path_argv(selected)]
            self._mutation_command(before, "unstage", args, worktree_id=worktree_id)
            return None, {"unstaged_paths": len(selected)}, selected

        return self._mutate(
            pid,
            "unstage",
            expected_state_token,
            dispatch,
            worktree_id=worktree_id,
            paths=selected,
        )

    def commit(
        self,
        pid: str,
        message: str,
        expected_state_token: str,
        *,
        amend: bool = False,
        worktree_id: str = "main",
    ) -> GitOperationResult:
        if not isinstance(message, str) or not message.strip() or "\x00" in message:
            raise ValidationError("Git commit message must be non-empty and contain no NUL")
        encoded = message.encode("utf-8")
        if len(encoded) > 131_072:
            raise ValidationError("Git commit message exceeds 131072 bytes")

        def dispatch(before: GitRepositoryState) -> tuple[str | None, dict[str, Any], Sequence[bytes]]:
            identity = self._run(
                ["var", "GIT_AUTHOR_IDENT"],
                worktree_id=worktree_id,
                max_output_bytes=65536,
            )
            if identity.returncode != 0:
                raise GitError(GitErrorCode.IDENTITY_MISSING.value, "Git commit identity is missing", operation="commit")
            args = ["commit", "--no-verify", "--no-gpg-sign", "-F", "-"]
            if amend:
                args.append("--amend")
            self._mutation_command(
                before,
                "commit",
                args,
                worktree_id=worktree_id,
                stdin=encoded,
            )
            current = self.provider.repository_state(worktree=self._worktree_path(worktree_id))
            return current.head_oid, {
                "amend": amend,
                "message_sha256": _sha256(encoded),
                "message_bytes": len(encoded),
            }, ()

        return self._mutate(
            pid,
            "commit",
            expected_state_token,
            dispatch,
            worktree_id=worktree_id,
            destructive=amend,
            extra={
                "amend": amend,
                "message_sha256": _sha256(encoded),
                "message_bytes": len(encoded),
            },
            protected_flow_snapshot_resolver=lambda _scopes: self._current_git_lineage_snapshot(
                expected_state_token,
                worktree_id=worktree_id,
                include_index=True,
                include_head=True,
            ),
        )

    def restore(
        self,
        pid: str,
        paths: Iterable[str | GitPath | dict[str, Any]],
        expected_state_token: str,
        *,
        staged: bool = False,
        worktree: bool = True,
        source: str | None = None,
        worktree_id: str = "main",
    ) -> GitOperationResult:
        selected = self._decode_paths(paths, required=True)
        if not staged and not worktree:
            raise ValidationError("Git restore must select staged and/or worktree content")
        if source is not None and worktree and not staged:
            raise GitError(
                GitErrorCode.UNSUPPORTED.value,
                "source-specific worktree-only restore requires Git newer than the supported minimum; select staged too",
            )

        def dispatch(before: GitRepositoryState) -> tuple[str | None, dict[str, Any], Sequence[bytes]]:
            oid = self._resolve_commit(source, worktree_id=worktree_id) if source is not None else before.head_oid
            if oid is None:
                raise GitError(GitErrorCode.INVALID_REF.value, "restore source is unavailable in an unborn repository")
            if staged:
                self._mark_git_source_carrier("commit", oid)
            else:
                self._mark_git_source_carrier("index", before.index_sha256)
            if staged and worktree:
                args = ["checkout", oid, "--", *self._path_argv(selected)]
            elif staged:
                args = ["reset", "-q", oid, "--", *self._path_argv(selected)]
            else:
                args = ["checkout", "--", *self._path_argv(selected)]
            self._mutation_command(before, "restore", args, worktree_id=worktree_id)
            return None, {"source_oid": oid, "staged": staged, "worktree": worktree}, selected

        return self._mutate(
            pid,
            "restore",
            expected_state_token,
            dispatch,
            worktree_id=worktree_id,
            paths=selected,
            filesystem_rights=(
                (CapabilityRight.WRITE, CapabilityRight.DELETE)
                if worktree
                else ()
            ),
            filesystem_path_subtree=True,
            destructive=True,
            extra={"staged": staged, "worktree": worktree, "source": source},
        )

    def branch(
        self,
        pid: str,
        action: str,
        name: str,
        expected_state_token: str,
        *,
        start: str | None = None,
        new_name: str | None = None,
        force: bool = False,
        worktree_id: str = "main",
    ) -> GitOperationResult:
        if action not in {"create", "delete", "rename"}:
            raise ValidationError("Git branch action must be create, delete, or rename")
        selected_name = self._validate_branch(name)
        selected_new = self._validate_branch(new_name) if new_name is not None else None
        if action == "rename" and selected_new is None:
            raise GitError(GitErrorCode.INVALID_REF.value, "branch rename requires new_name")
        destructive = action in {"delete", "rename"} or force

        def dispatch(before: GitRepositoryState) -> tuple[str | None, dict[str, Any], Sequence[bytes]]:
            created_oid: str | None = None
            if action == "create":
                oid = self._resolve_commit(start, worktree_id=worktree_id) if start is not None else before.head_oid
                if oid is None:
                    raise GitError(GitErrorCode.INVALID_REF.value, "branch start is unavailable")
                args = ["branch", "-f" if force else "--no-track", selected_name, oid]
                self._mutation_command(before, "branch", args, worktree_id=worktree_id)
                created_oid = oid
            elif action == "delete":
                args = ["branch", "-D" if force else "-d", "--", selected_name]
                self._mutation_command(before, "branch", args, worktree_id=worktree_id)
            else:
                assert selected_new is not None
                args = ["branch", "-M" if force else "-m", "--", selected_name, selected_new]
                self._mutation_command(before, "branch", args, worktree_id=worktree_id)
            return created_oid, {
                "action": action,
                "branch": selected_name,
                "new_branch": selected_new,
                "force": force,
            }, ()

        return self._mutate(
            pid,
            "branch",
            expected_state_token,
            dispatch,
            worktree_id=worktree_id,
            destructive=destructive,
            extra={"action": action, "branch": selected_name, "new_branch": selected_new, "force": force},
        )

    def switch(
        self,
        pid: str,
        target: str,
        expected_state_token: str,
        *,
        create: bool = False,
        start: str | None = None,
        detach: bool = False,
        force: bool = False,
        worktree_id: str = "main",
    ) -> GitOperationResult:
        if create and detach:
            raise ValidationError("Git switch cannot both create a branch and detach")
        selected_branch = self._validate_branch(target) if not detach else None

        def dispatch(before: GitRepositoryState) -> tuple[str | None, dict[str, Any], Sequence[bytes]]:
            args = ["checkout"]
            if force:
                args.append("--force")
            created_oid: str | None = None
            if detach:
                oid = self._resolve_commit(target, worktree_id=worktree_id)
                self._mark_git_source_carrier("commit", oid)
                args.extend(["--detach", oid])
                created_oid = oid
            elif create:
                oid = self._resolve_commit(start, worktree_id=worktree_id) if start is not None else before.head_oid
                if oid is None:
                    raise GitError(GitErrorCode.INVALID_REF.value, "new branch start is unavailable")
                self._mark_git_source_carrier("commit", oid)
                assert selected_branch is not None
                args.extend(["-b", selected_branch, oid])
                created_oid = oid
            else:
                assert selected_branch is not None
                oid = self._resolve_commit(selected_branch, worktree_id=worktree_id)
                self._mark_git_source_carrier("commit", oid)
                args.append(selected_branch)
            self._mutation_command(before, "switch", args, worktree_id=worktree_id)
            return created_oid, {
                "branch": selected_branch,
                "detached_oid": created_oid if detach else None,
                "created": create,
                "force": force,
            }, ()

        return self._mutate(
            pid,
            "switch",
            expected_state_token,
            dispatch,
            worktree_id=worktree_id,
            filesystem_rights=(
                CapabilityRight.READ,
                CapabilityRight.WRITE,
                CapabilityRight.DELETE,
            ),
            destructive=force,
            extra={"target": target, "create": create, "start": start, "detach": detach, "force": force},
        )

    def tag(
        self,
        pid: str,
        action: str,
        name: str,
        expected_state_token: str,
        *,
        target: str | None = None,
        message: str | None = None,
        force: bool = False,
        worktree_id: str = "main",
    ) -> GitOperationResult:
        if action not in {"create", "delete"}:
            raise ValidationError("Git tag action must be create or delete")
        selected_name = self._validate_branch(name)
        encoded_message = message.encode("utf-8") if message is not None else None
        if encoded_message is not None and (not message.strip() or b"\x00" in encoded_message or len(encoded_message) > 131_072):
            raise ValidationError("Git tag message is invalid")
        destructive = action == "delete" or force

        def dispatch(before: GitRepositoryState) -> tuple[str | None, dict[str, Any], Sequence[bytes]]:
            if action == "delete":
                self._mutation_command(before, "tag", ["tag", "-d", "--", selected_name], worktree_id=worktree_id)
                oid = None
            else:
                oid = self._resolve_commit(target, worktree_id=worktree_id) if target is not None else before.head_oid
                if oid is None:
                    raise GitError(GitErrorCode.INVALID_REF.value, "tag target is unavailable")
                args = ["tag"]
                if force:
                    args.append("-f")
                if encoded_message is not None:
                    args.extend(["-a", "--no-sign", "-F", "-"])
                args.extend([selected_name, oid])
                self._mutation_command(
                    before,
                    "tag",
                    args,
                    worktree_id=worktree_id,
                    stdin=encoded_message,
                )
            return oid, {
                "action": action,
                "tag": selected_name,
                "force": force,
                "message_sha256": _sha256(encoded_message) if encoded_message is not None else None,
            }, ()

        return self._mutate(
            pid,
            "tag",
            expected_state_token,
            dispatch,
            worktree_id=worktree_id,
            destructive=destructive,
            extra={
                "action": action,
                "tag": selected_name,
                "target": target,
                "force": force,
                "message_sha256": _sha256(encoded_message) if encoded_message is not None else None,
            },
        )

    def integrate(
        self,
        pid: str,
        operation: str,
        expected_state_token: str,
        *,
        ref: str | None = None,
        abort_kind: str | None = None,
        worktree_id: str = "main",
    ) -> GitOperationResult:
        if operation not in {"merge", "rebase", "cherry_pick", "revert", "abort"}:
            raise ValidationError("Git integrate operation is invalid")
        if operation == "abort":
            if abort_kind not in {"merge", "rebase", "cherry_pick", "revert"}:
                raise ValidationError("Git abort requires merge/rebase/cherry_pick/revert abort_kind")
            if ref is not None:
                raise ValidationError("Git abort does not accept a ref")
        elif ref is None:
            raise GitError(GitErrorCode.INVALID_REF.value, "Git integrate requires a ref")
        destructive = operation in {"rebase", "abort"}

        def dispatch(before: GitRepositoryState) -> tuple[str | None, dict[str, Any], Sequence[bytes]]:
            oid: str | None = None
            if operation == "abort":
                command_name = "cherry-pick" if abort_kind == "cherry_pick" else str(abort_kind)
                args = [command_name, "--abort"]
            else:
                assert ref is not None
                oid = self._resolve_commit(ref, worktree_id=worktree_id)
                self._mark_git_source_carrier("commit", oid)
                if operation == "merge":
                    previous_stash_oid = self._resolve_stash_commit(
                        0,
                        worktree_id=worktree_id,
                        optional=True,
                    )
                    args = [
                        "merge",
                        "--no-edit",
                        "--no-gpg-sign",
                        "--commit",
                        "--no-squash",
                        oid,
                    ]
                elif operation == "rebase":
                    args = ["rebase", "--no-autostash", oid]
                elif operation == "cherry_pick":
                    args = ["cherry-pick", "--no-gpg-sign", oid]
                else:
                    args = ["revert", "--no-edit", "--no-gpg-sign", oid]
            self._mutation_command(before, "integrate", args, worktree_id=worktree_id)
            current = self.provider.repository_state(worktree=self._worktree_path(worktree_id))
            if operation == "merge":
                merge_head = self._run(
                    [
                        "rev-parse",
                        "--verify",
                        "--quiet",
                        "--end-of-options",
                        "MERGE_HEAD",
                    ],
                    worktree_id=worktree_id,
                    max_output_bytes=65536,
                )
                if merge_head.returncode not in {0, 1}:
                    self._require_success(merge_head, "integrate")
                current_stash_oid = self._resolve_stash_commit(
                    0,
                    worktree_id=worktree_id,
                    optional=True,
                )
                parsed = self._parse_status(
                    current,
                    limit=self.config.git.status_entry_hard_limit,
                )
                non_terminal = (
                    merge_head.returncode == 0
                    or any(
                        entry.kind == GitStatusKind.UNMERGED
                        for entry in parsed.entries
                    )
                    or current_stash_oid != previous_stash_oid
                )
                if non_terminal:
                    raise GitError(
                        GitErrorCode.CONFLICT.value,
                        "typed Git merge returned without a terminal, stash-preserving state",
                        operation="integrate",
                    )
                if current.head_oid is None:
                    raise GitError(
                        GitErrorCode.COMMAND_FAILED.value,
                        "typed Git merge returned without a final HEAD",
                        operation="integrate",
                    )
                ancestry = self._run(
                    [
                        "merge-base",
                        "--is-ancestor",
                        oid,
                        current.head_oid,
                    ],
                    worktree_id=worktree_id,
                    max_output_bytes=65536,
                )
                if ancestry.returncode not in {0, 1}:
                    self._require_success(ancestry, "integrate")
                if ancestry.returncode != 0:
                    raise GitError(
                        GitErrorCode.COMMAND_FAILED.value,
                        "typed Git merge did not integrate the selected commit",
                        operation="integrate",
                    )
            return current.head_oid, {"integration": operation, "source_oid": oid, "abort_kind": abort_kind}, ()

        return self._mutate(
            pid,
            "integrate",
            expected_state_token,
            dispatch,
            worktree_id=worktree_id,
            filesystem_rights=(
                CapabilityRight.READ,
                CapabilityRight.WRITE,
                CapabilityRight.DELETE,
            ),
            destructive=destructive,
            extra={"integration": operation, "ref": ref, "abort_kind": abort_kind},
        )

    def stash(
        self,
        pid: str,
        action: str,
        expected_state_token: str,
        *,
        index: int = 0,
        include_untracked: bool = False,
        reinstate_index: bool = False,
        worktree_id: str = "main",
    ) -> GitOperationResult:
        if action not in {"push", "apply", "pop", "drop", "clear"}:
            raise ValidationError("Git stash action must be push/apply/pop/drop/clear")
        if isinstance(index, bool) or index < 0 or index > 100_000:
            raise ValidationError("Git stash index is invalid")
        destructive = action in {"pop", "drop", "clear"} or (
            action == "push" and include_untracked
        )
        stash_ref = f"stash@{{{index}}}"

        def dispatch(before: GitRepositoryState) -> tuple[str | None, dict[str, Any], Sequence[bytes]]:
            previous_stash_oid: str | None = None
            if action == "push":
                previous_stash_oid = self._resolve_stash_commit(
                    0,
                    worktree_id=worktree_id,
                    optional=True,
                )
                if self.data_flow is None:
                    raise ValidationError("Git data-flow manager is not attached")
                worktree_root = self._normalized_worktree_root(worktree_id)
                source_context, _state_version = self.data_flow.file_tree_snapshot(
                    worktree_root or "."
                )
                self._mark_git_source_context(source_context)
                args = ["stash", "push", "--message", "agent-libos"]
                if include_untracked:
                    args.append("--include-untracked")
            elif action in {"apply", "pop"}:
                stash_oid = self._resolve_stash_commit(
                    index,
                    worktree_id=worktree_id,
                )
                assert stash_oid is not None
                self._mark_git_source_carrier("commit", stash_oid)
                args = ["stash", action]
                if reinstate_index:
                    args.append("--index")
                args.append(stash_ref)
            elif action == "drop":
                args = ["stash", "drop", stash_ref]
            else:
                args = ["stash", "clear"]
            self._mutation_command(before, "stash", args, worktree_id=worktree_id)
            current_stash_oid = (
                self._resolve_stash_commit(
                    0,
                    worktree_id=worktree_id,
                    optional=True,
                )
                if action == "push"
                else None
            )
            created_oid = (
                current_stash_oid
                if current_stash_oid != previous_stash_oid
                else None
            )
            return created_oid, {
                "action": action,
                "stash_index": index if action != "clear" else None,
                "include_untracked": include_untracked,
                "reinstate_index": reinstate_index,
            }, ()

        filesystem_rights: tuple[CapabilityRight, ...] = (
            (CapabilityRight.READ, CapabilityRight.WRITE, CapabilityRight.DELETE)
            if action in {"push", "apply", "pop"}
            else ()
        )
        return self._mutate(
            pid,
            "stash",
            expected_state_token,
            dispatch,
            worktree_id=worktree_id,
            filesystem_rights=filesystem_rights,
            destructive=destructive,
            extra={
                "action": action,
                "stash_index": index,
                "include_untracked": include_untracked,
                "reinstate_index": reinstate_index,
            },
        )

    def reset(
        self,
        pid: str,
        target: str,
        expected_state_token: str,
        *,
        mode: str = "mixed",
        worktree_id: str = "main",
    ) -> GitOperationResult:
        if mode not in {"soft", "mixed", "hard"}:
            raise ValidationError("Git reset mode must be soft, mixed, or hard")

        def dispatch(before: GitRepositoryState) -> tuple[str | None, dict[str, Any], Sequence[bytes]]:
            oid = self._resolve_commit(target, worktree_id=worktree_id)
            self._mark_git_source_carrier("commit", oid)
            self._mutation_command(before, "reset", ["reset", f"--{mode}", oid], worktree_id=worktree_id)
            return oid, {"mode": mode, "target_oid": oid, "old_head_oid": before.head_oid}, ()

        fs_rights = (
            (CapabilityRight.READ, CapabilityRight.WRITE, CapabilityRight.DELETE)
            if mode == "hard"
            else ()
        )
        return self._mutate(
            pid,
            "reset",
            expected_state_token,
            dispatch,
            worktree_id=worktree_id,
            filesystem_rights=fs_rights,
            destructive=True,
            extra={"mode": mode, "target": target},
        )

    def _clean_snapshot(
        self,
        paths: Sequence[bytes],
        *,
        directories: bool,
        ignored: bool,
        worktree_id: str,
    ) -> _GitCleanSnapshot:
        flags = "-f"
        if directories:
            flags += "d"
        if ignored:
            flags += "x"
        suffix = ["--", *self._path_argv(paths)]
        preview = self._require_success(
            self._run(
                ["clean", flags.replace("f", "n", 1), *suffix],
                worktree_id=worktree_id,
                max_output_bytes=self.config.git.output_hard_limit_bytes,
            ),
            "clean_preview",
        )
        candidate_commands = [
            ["ls-files", "--others", "--exclude-standard", "-z", *suffix]
        ]
        if ignored:
            candidate_commands.append(
                [
                    "ls-files",
                    "--others",
                    "--ignored",
                    "--exclude-standard",
                    "-z",
                    *suffix,
                ]
            )
        candidates: dict[bytes, None] = {}
        for command in candidate_commands:
            listed = self._require_success(
                self._run(
                    command,
                    worktree_id=worktree_id,
                    max_output_bytes=self.config.git.output_hard_limit_bytes,
                ),
                "clean_candidates",
            )
            for path in self._bounded_git_paths(listed.stdout):
                candidates.setdefault(self._decode_path(path), None)

        digest = hashlib.sha256()
        digest.update(len(preview.stdout).to_bytes(8, "big"))
        digest.update(preview.stdout)
        selected_paths = tuple(sorted(candidates))
        for raw_path in selected_paths:
            content_sha256 = self.provider.path_content_sha256(
                raw_path,
                worktree=self._worktree_path(worktree_id),
            )
            kind = self.provider.path_kind(
                raw_path,
                worktree=self._worktree_path(worktree_id),
            )
            digest.update(len(raw_path).to_bytes(8, "big"))
            digest.update(raw_path)
            digest.update(kind.encode("ascii"))
            digest.update((content_sha256 or "missing").encode("ascii"))
        return _GitCleanSnapshot(
            flags=flags,
            preview_sha256=preview.stdout_sha256,
            preview_bytes=len(preview.stdout),
            candidate_paths=selected_paths,
            manifest_sha256=digest.hexdigest(),
        )

    def clean(
        self,
        pid: str,
        expected_state_token: str,
        *,
        paths: Iterable[str | GitPath | dict[str, Any]] = (),
        directories: bool = False,
        ignored: bool = False,
        worktree_id: str = "main",
    ) -> GitOperationResult:
        selected = self._decode_paths(paths)

        def read_snapshot() -> tuple[_GitCleanSnapshot, dict[str, Any]]:
            current = self._clean_snapshot(
                selected,
                directories=directories,
                ignored=ignored,
                worktree_id=worktree_id,
            )
            return current, {
                "candidate_count": len(current.candidate_paths),
                "candidate_manifest_sha256": current.manifest_sha256,
                "preview_bytes": current.preview_bytes,
                "preview_sha256": current.preview_sha256,
            }

        snapshot = self._read(
            pid,
            "clean_snapshot",
            read_snapshot,
            worktree_id=worktree_id,
            extra={"directories": directories, "ignored": ignored},
        )

        def dispatch(before: GitRepositoryState) -> tuple[str | None, dict[str, Any], Sequence[bytes]]:
            current = self._clean_snapshot(
                selected,
                directories=directories,
                ignored=ignored,
                worktree_id=worktree_id,
            )
            if current.manifest_sha256 != snapshot.manifest_sha256:
                raise GitError(
                    GitErrorCode.STALE_STATE.value,
                    "Git clean candidates changed after approval",
                    retryable=True,
                )
            suffix = ["--", *self._path_argv(selected)]
            self._mutation_command(
                before,
                "clean",
                ["clean", snapshot.flags, *suffix],
                worktree_id=worktree_id,
            )
            return None, {
                "directories": directories,
                "ignored": ignored,
                "preview_sha256": snapshot.preview_sha256,
                "preview_bytes": snapshot.preview_bytes,
                "candidate_manifest_sha256": snapshot.manifest_sha256,
                "candidate_count": len(snapshot.candidate_paths),
            }, snapshot.candidate_paths or selected

        return self._mutate(
            pid,
            "clean",
            expected_state_token,
            dispatch,
            worktree_id=worktree_id,
            paths=selected,
            reconciliation_paths=snapshot.candidate_paths,
            filesystem_rights=(CapabilityRight.READ, CapabilityRight.DELETE),
            destructive=True,
            extra={
                "directories": directories,
                "ignored": ignored,
                "preview_sha256": snapshot.preview_sha256,
                "preview_bytes": snapshot.preview_bytes,
                "candidate_manifest_sha256": snapshot.manifest_sha256,
                "candidate_count": len(snapshot.candidate_paths),
            },
        )

    def worktree(
        self,
        pid: str,
        action: str,
        expected_state_token: str,
        *,
        ref: str | None = None,
        new_branch: str | None = None,
        managed_worktree_id: str | None = None,
    ) -> GitOperationResult:
        if action not in {"create", "remove"}:
            raise ValidationError("Git worktree action must be create or remove")
        if action == "create":
            if managed_worktree_id is not None:
                raise GitError(GitErrorCode.INVALID_PATH.value, "Runtime generates managed worktree ids")
            selected_branch = self._validate_branch(new_branch) if new_branch is not None else None
            selected_expected_state = self._validate_expected_token(
                expected_state_token
            )
            selected_id = self._stable_generated_id(
                "wt",
                pid,
                selected_expected_state,
                ref,
                selected_branch,
            )
            if not _WORKTREE_ID_RE.fullmatch(selected_id):
                raise RuntimeError("generated worktree id is invalid")
        else:
            if ref is not None or new_branch is not None:
                raise ValidationError("Git worktree remove accepts only managed_worktree_id")
            if managed_worktree_id is None or not _WORKTREE_ID_RE.fullmatch(managed_worktree_id) or managed_worktree_id == "main":
                raise GitError(GitErrorCode.INVALID_PATH.value, "invalid managed worktree id")
            selected_id = managed_worktree_id
            selected_branch = None
            selected_expected_state = self._validate_expected_token(
                expected_state_token
            )
        workspace_root = Path(getattr(self.provider, "workspace_root", self.filesystem.root))
        managed_path = Path(getattr(self.provider, "managed_worktree_root", workspace_root)) / selected_id
        try:
            relative_path = managed_path.relative_to(workspace_root).as_posix()
        except ValueError as exc:
            raise GitError(GitErrorCode.INVALID_PATH.value, "managed worktree path escaped workspace") from exc
        fs_path = os.fsencode(relative_path)

        def dispatch(before: GitRepositoryState) -> tuple[str | None, dict[str, Any], Sequence[bytes]]:
            if action == "create":
                self._mark_git_effect_started()
                target = self.provider.prepare_managed_worktree(selected_id)
                oid = self._resolve_commit(ref, worktree_id="main") if ref is not None else before.head_oid
                if oid is None:
                    raise GitError(GitErrorCode.INVALID_REF.value, "managed worktree start is unavailable")
                self._mark_git_source_carrier("commit", oid)
                if selected_branch is None:
                    args = ["worktree", "add", "--detach", str(target), oid]
                else:
                    args = ["worktree", "add", "-b", selected_branch, str(target), oid]
                try:
                    self._mutation_command(before, "worktree", args, worktree_id="main")
                except BaseException:
                    with contextlib.suppress(GitError):
                        self.provider.repository_layout(worktree=target)
                        self._mark_git_materialized_worktree(selected_id)
                    raise
                self._mark_git_materialized_worktree(selected_id)
                self.provider.repository_layout(worktree=target)
                created_oid = oid
            else:
                target = Path(self._worktree_path(selected_id) or "")
                self.provider.repository_layout(worktree=target)
                self._mutation_command(
                    before,
                    "worktree",
                    ["worktree", "remove", "--", str(target)],
                    worktree_id="main",
                )
                created_oid = None
            return created_oid, {
                "action": action,
                "managed_worktree_id": selected_id,
                "branch": selected_branch,
            }, (fs_path,)

        return self._mutate(
            pid,
            "worktree",
            selected_expected_state,
            dispatch,
            worktree_id="main",
            paths=(fs_path,),
            materialized_worktree_id=(
                selected_id if action == "create" else None
            ),
            filesystem_rights=(
                (CapabilityRight.WRITE,)
                if action == "create"
                else (CapabilityRight.DELETE,)
            ),
            filesystem_path_subtree=True,
            destructive=action == "remove",
            extra={
                "action": action,
                "managed_worktree_id": selected_id,
                "ref": ref,
                "branch": selected_branch,
            },
        )

    def _index_tree_oid(self, *, worktree_id: str) -> str:
        layout = self.provider.repository_layout(worktree=self._worktree_path(worktree_id))
        result = self._require_success(
            self._run(
                ["ls-files", "--stage", "-z"],
                worktree_id=worktree_id,
                max_output_bytes=self.config.git.output_hard_limit_bytes,
            ),
            "create_patch",
        )
        tree: dict[bytes, Any] = {}
        oid_bytes = 20 if layout.object_format == "sha1" else 32
        for record in result.stdout.split(b"\0"):
            if not record:
                continue
            metadata, separator, path = record.partition(b"\t")
            fields = metadata.split(b" ")
            if not separator or len(fields) != 3:
                raise GitError(GitErrorCode.COMMAND_FAILED.value, "Git index output is malformed")
            mode, oid_hex, stage = fields
            if stage != b"0":
                raise GitError(GitErrorCode.CONFLICT.value, "conflicted index cannot produce a patch artifact")
            if mode not in {b"100644", b"100755", b"120000", b"160000"}:
                raise GitError(GitErrorCode.COMMAND_FAILED.value, "Git index contains an unsupported mode")
            try:
                object_id = bytes.fromhex(oid_hex.decode("ascii"))
            except (ValueError, UnicodeDecodeError) as exc:
                raise GitError(GitErrorCode.COMMAND_FAILED.value, "Git index contains an invalid OID") from exc
            if len(object_id) != oid_bytes:
                raise GitError(GitErrorCode.COMMAND_FAILED.value, "Git index OID length is invalid")
            if not any(object_id):
                raise GitError(
                    GitErrorCode.CONFLICT.value,
                    "intent-to-add index entries cannot produce a patch artifact",
                )
            parts = path.split(b"/")
            if any(part in {b"", b".", b".."} for part in parts):
                raise GitError(GitErrorCode.COMMAND_FAILED.value, "Git index contains an unsafe path")
            node = tree
            for part in parts[:-1]:
                existing = node.get(part)
                if existing is None:
                    existing = {}
                    node[part] = existing
                if not isinstance(existing, dict):
                    raise GitError(GitErrorCode.COMMAND_FAILED.value, "Git index path hierarchy is invalid")
                node = existing
            if parts[-1] in node:
                raise GitError(GitErrorCode.COMMAND_FAILED.value, "Git index contains duplicate paths")
            node[parts[-1]] = (mode, object_id)

        def hash_tree(node: dict[bytes, Any]) -> bytes:
            content = bytearray()
            for name, value in sorted(
                node.items(),
                key=lambda item: item[0] + (b"/" if isinstance(item[1], dict) else b""),
            ):
                if isinstance(value, dict):
                    mode = b"40000"
                    child_oid = hash_tree(value)
                else:
                    mode, child_oid = value
                content.extend(mode)
                content.extend(b" ")
                content.extend(name)
                content.extend(b"\0")
                content.extend(child_oid)
            header = b"tree " + str(len(content)).encode("ascii") + b"\0"
            digest = hashlib.new(layout.object_format)
            digest.update(header)
            digest.update(content)
            return digest.digest()

        return hash_tree(tree).hex()

    def create_patch(
        self,
        pid: str,
        *,
        scope: str = "worktree",
        base: str | None = None,
        head: str | None = None,
        paths: Iterable[str | GitPath | dict[str, Any]] = (),
        worktree_id: str = "main",
    ) -> GitPatchArtifact:
        if self.memory is None:
            raise GitError(GitErrorCode.UNSUPPORTED.value, "Git patch artifacts require Object Memory")
        selected_paths = self._decode_paths(paths)

        def read() -> tuple[
            tuple[GitDiffResult, bytes, str],
            dict[str, Any],
        ]:
            return self._patch_material(
                scope=scope,
                base=base,
                head=head,
                paths=selected_paths,
                worktree_id=worktree_id,
            )

        result, patch, index_oid = self._read(
            pid,
            "create_patch",
            read,
            right=CapabilityRight.DIFF,
            worktree_id=worktree_id,
            extra={
                "scope": scope,
                "base": base,
                "head": head,
                "path_count": len(selected_paths),
                "paths_sha256": _sha256(b"\0".join(selected_paths)),
            },
            flow_snapshot_resolver=lambda: self._diff_read_flow_snapshot(
                scope=scope,
                base=base,
                head=head,
                paths=selected_paths,
                worktree_id=worktree_id,
                include_patch_index=scope == "range",
            ),
            flow_lock_paths=(
                tuple(
                    self._normalized_file_path(path, worktree_id=worktree_id)
                    for path in selected_paths
                )
                or (self._normalized_worktree_root(worktree_id) or ".",)
                if scope == "worktree"
                else ()
            ),
        )
        return self._store_patch_artifact(
            pid=pid,
            result=result,
            patch=patch,
            index_oid=index_oid,
            worktree_id=worktree_id,
        )

    def _patch_material(
        self,
        *,
        scope: str,
        base: str | None,
        head: str | None,
        paths: Sequence[bytes],
        worktree_id: str,
    ) -> tuple[tuple[GitDiffResult, bytes, str], dict[str, Any]]:
        result, summary = self._diff_result(
            scope=scope,
            base=base,
            head=head,
            paths=paths,
            worktree_id=worktree_id,
            selected_limit=self.config.git.patch_hard_limit_bytes,
            hard_limit=self.config.git.patch_hard_limit_bytes,
            operation="create_patch",
        )
        if result.truncated or result.bytes > self.config.git.patch_max_bytes:
            raise GitError(
                GitErrorCode.OUTPUT_TOO_LARGE.value,
                "Git patch exceeds the configured artifact limit",
                operation="create_patch",
            )
        patch = base64.b64decode(result.patch_b64, validate=True)
        index_oid = self._index_tree_oid(worktree_id=worktree_id)
        current = self.provider.repository_state(
            worktree=self._worktree_path(worktree_id)
        )
        if self._state_token(current).token != result.state.token:
            raise GitError(
                GitErrorCode.STALE_STATE.value,
                "Git state changed before patch artifact creation",
                operation="create_patch",
                retryable=True,
            )
        return (result, patch, index_oid), {
            **summary,
            "index_oid": index_oid,
            "patch_sha256": _sha256(patch),
            "patch_bytes": len(patch),
        }

    def _store_patch_artifact(
        self,
        *,
        pid: str,
        result: GitDiffResult,
        patch: bytes,
        index_oid: str,
        worktree_id: str,
    ) -> GitPatchArtifact:
        if self.memory is None:
            raise GitError(
                GitErrorCode.UNSUPPORTED.value,
                "Git patch artifacts require Object Memory",
            )
        payload = {
            "artifact_schema": "agent-libos.git-patch.v1",
            "repository_id": result.repository_id,
            "worktree_id": result.worktree_id,
            "base_oid": result.base_oid or result.state.head_oid,
            "head_oid": result.head_oid or result.state.head_oid,
            "index_oid": index_oid,
            "state": asdict(result.state),
            "changed_paths": [asdict(path) for path in result.changed_paths],
            "patch_b64": result.patch_b64,
            "patch_bytes": len(patch),
            "patch_sha256": _sha256(patch),
        }
        encoded_payload = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(encoded_payload) > self.config.tools.memory_payload_hard_limit_bytes:
            raise GitError(
                GitErrorCode.OUTPUT_TOO_LARGE.value,
                "Git patch exceeds the Object Memory payload limit",
                operation="create_patch",
            )
        artifact_context = DataFlowContext()
        artifact_parent_oids: tuple[str, ...] = ()
        artifact_source_refs: tuple[str, ...] = ()
        if self.data_flow is not None:
            artifact_context = DataFlowContext.aggregate(
                [
                    self.data_flow.current_context(),
                    *(
                        self.data_flow.file_context(
                            self._normalized_file_path(
                                self._decode_path(path),
                                worktree_id=worktree_id,
                            )
                        )
                        for path in result.changed_paths
                    ),
                ]
            )
            artifact_parent_oids, artifact_source_refs = (
                self.data_flow.provenance_sources(artifact_context)
            )
        handle = self.memory.create_object(
            pid,
            ObjectType.CODE_PATCH,
            payload,
            metadata=ObjectMetadata(
                title="Immutable Git patch artifact",
                summary=f"Git patch {_sha256(patch)[:16]}",
                tags=["git", "patch", "immutable"],
                mime_type="text/x-diff",
                **artifact_context.labels.to_dict(),
            ),
            immutable=True,
            provenance=Provenance(
                parent_oids=list(artifact_parent_oids),
                source_refs=[
                    *artifact_source_refs,
                    f"git:{result.repository_id}:{result.state.token}",
                ],
                created_from_action="primitive.git.create_patch",
            ),
            name=f"git_patch_{_sha256(patch)[:16]}_{new_id('artifact')[-8:]}",
        )
        return GitPatchArtifact(
            oid=handle.oid,
            repository_id=result.repository_id,
            worktree_id=result.worktree_id,
            base_oid=payload["base_oid"],
            head_oid=payload["head_oid"],
            index_oid=index_oid,
            patch_sha256=payload["patch_sha256"],
            bytes=len(patch),
            changed_paths=result.changed_paths,
            state=result.state,
        )

    def _load_patch_artifact(self, pid: str, patch_oid: str) -> tuple[Any, dict[str, Any], bytes, tuple[bytes, ...]]:
        if self.memory is None:
            raise GitError(GitErrorCode.UNSUPPORTED.value, "Git patch artifacts require Object Memory")
        handle = self.memory.handle_for_oid(
            pid,
            patch_oid,
            required_rights=[ObjectRight.READ],
            issued_by="git.patch",
        )
        obj = self.memory.get_object(pid, handle)
        if obj.type is not ObjectType.CODE_PATCH or obj.immutable is not True:
            raise GitError(GitErrorCode.UNSUPPORTED.value, "object is not an immutable code patch")
        if obj.provenance.created_from_action != "primitive.git.create_patch":
            raise GitError(GitErrorCode.UNSAFE_REPOSITORY.value, "patch object lacks trusted Git artifact provenance")
        payload = obj.payload
        if not isinstance(payload, dict) or payload.get("artifact_schema") != "agent-libos.git-patch.v1":
            raise GitError(GitErrorCode.UNSUPPORTED.value, "patch artifact schema is unsupported")
        try:
            patch = base64.b64decode(str(payload["patch_b64"]), validate=True)
            expected_bytes = int(payload["patch_bytes"])
            expected_sha256 = str(payload["patch_sha256"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GitError(GitErrorCode.UNSUPPORTED.value, "patch artifact payload is malformed") from exc
        if len(patch) != expected_bytes or _sha256(patch) != expected_sha256:
            raise GitError(GitErrorCode.UNSAFE_REPOSITORY.value, "patch artifact integrity check failed")
        if len(patch) > self.config.git.patch_max_bytes:
            raise GitError(GitErrorCode.OUTPUT_TOO_LARGE.value, "patch artifact exceeds the configured limit")
        raw_paths: list[bytes] = []
        for value in payload.get("changed_paths", []):
            raw_paths.append(self._decode_path(value))
        if not raw_paths and patch:
            raise GitError(GitErrorCode.UNSAFE_REPOSITORY.value, "non-empty patch artifact has no path manifest")
        return obj, payload, patch, tuple(dict.fromkeys(raw_paths))

    def _normalized_worktree_root(self, worktree_id: str) -> str:
        if worktree_id == "main":
            return ""
        root = Path(getattr(self.provider, "workspace_root", self.filesystem.root))
        selected = Path(self._worktree_path(worktree_id) or "")
        return selected.relative_to(root).as_posix()

    def _normalized_file_path(self, raw_path: bytes, *, worktree_id: str) -> str:
        suffix = os.fsdecode(raw_path).replace(os.sep, "/")
        prefix = self._normalized_worktree_root(worktree_id)
        return f"{prefix}/{suffix}".strip("/")

    def apply_patch(
        self,
        pid: str,
        patch_oid: str,
        expected_state_token: str,
        *,
        index: bool = False,
        worktree_id: str = "main",
    ) -> GitOperationResult:
        _obj, payload, patch, paths = self._load_patch_artifact(pid, patch_oid)
        deletes = b"\ndeleted file mode " in b"\n" + patch or b"\n+++ /dev/null" in b"\n" + patch
        source_context = (
            self.data_flow.context_from_source_oids(pid, [patch_oid])
            if self.data_flow is not None
            else None
        )

        def dispatch(before: GitRepositoryState) -> tuple[str | None, dict[str, Any], Sequence[bytes]]:
            if payload.get("repository_id") != before.layout.repository_id:
                raise GitError(GitErrorCode.UNSAFE_REPOSITORY.value, "patch artifact belongs to a different repository")
            args = ["apply", "--whitespace=error-all", "--recount"]
            if index:
                args.append("--index")
            check = self._run(
                [*args, "--check", "-"],
                worktree_id=worktree_id,
                max_output_bytes=self.config.git.output_hard_limit_bytes,
                stdin=patch,
            )
            self._require_success(check, "apply_patch_preview")
            self._mutation_command(
                before,
                "apply_patch",
                [*args, "-"],
                worktree_id=worktree_id,
                stdin=patch,
            )
            if self.data_flow is not None and source_context is not None:
                with self.data_flow.store.transaction():
                    for raw_path in paths:
                        normalized = self._normalized_file_path(raw_path, worktree_id=worktree_id)
                        digest = self.provider.path_content_sha256(
                            raw_path,
                            worktree=self._worktree_path(worktree_id),
                        )
                        if digest is None:
                            previous = self.data_flow.store.get_file_label_binding(normalized)
                            if previous is not None:
                                self.data_flow.tombstone_file(
                                    pid=pid,
                                    normalized_path=normalized,
                                    expected_binding_id=previous.binding_id,
                                    expected_generation=previous.generation,
                                )
                        else:
                            self.data_flow.bind_written_file_digest(
                                pid=pid,
                                normalized_path=normalized,
                                content_sha256=digest,
                                context=source_context,
                            )
            return None, {
                "patch_oid": patch_oid,
                "patch_sha256": _sha256(patch),
                "patch_bytes": len(patch),
                "index": index,
                "deletes_paths": deletes,
            }, paths

        return self._mutate(
            pid,
            "apply_patch",
            expected_state_token,
            dispatch,
            worktree_id=worktree_id,
            paths=paths,
            filesystem_rights=(
                (CapabilityRight.READ, CapabilityRight.WRITE, CapabilityRight.DELETE)
                if deletes
                else (CapabilityRight.READ, CapabilityRight.WRITE)
            ),
            destructive=deletes,
            extra={
                "patch_oid": patch_oid,
                "patch_sha256": _sha256(patch),
                "patch_bytes": len(patch),
                "index": index,
                "deletes_paths": deletes,
                "base_oid": payload.get("base_oid"),
                "head_oid": payload.get("head_oid"),
                "index_oid": payload.get("index_oid"),
            },
            source_oids=[patch_oid],
            input_refs=[patch_oid],
        )

    def _remote_preflight(
        self,
        pid: str,
        remote: str,
        right: CapabilityRight,
        *,
        worktree_id: str,
        effect_operation: str,
    ) -> dict[str, Any]:
        selected = self._validate_remote(remote)
        if effect_operation not in {"fetch", "push"}:
            raise ValidationError("invalid Git remote effect operation")
        resource = self.remote_resource(selected)
        self.protected_operations.authority_policy.assert_effect(
            pid,
            f"git.{effect_operation}",
        )
        matching = self.capabilities.matching_capabilities(
            pid,
            resource,
            right,
            include_ask=True,
        )
        has_deny = any(
            capability.effect is CapabilityEffect.DENY
            for capability in matching
        )
        has_candidate = any(
            capability.effect in {CapabilityEffect.ALLOW, CapabilityEffect.ASK}
            for capability in matching
        )
        if has_deny or not has_candidate:
            self.capabilities.require(
                pid,
                resource,
                right,
                {
                    "primitive": "runtime.git.remote_preflight",
                    "operation": "remote_preflight",
                    "resource": resource,
                    "right": right.value,
                    "git_remote": selected,
                },
                consume=False,
            )
        with self._provider_subprocess_scope(
            pid,
            "remote_preflight",
            remote=selected,
        ):
            return self.provider.preflight_remote_fingerprint(
                selected,
                worktree=self._worktree_path(worktree_id),
            )

    @staticmethod
    def _fingerprint_context(
        fingerprint: dict[str, Any],
        *,
        remote: str,
        use_push_url: bool,
        remote_ref: str | None = None,
        old_oid: str | None = None,
    ) -> dict[str, Any]:
        return {
            "git_remote": remote,
            "git_remote_ref": remote_ref,
            "git_url_fingerprint": fingerprint[
                "push_url_sha256" if use_push_url else "fetch_url_sha256"
            ],
            "git_remote_fingerprint": fingerprint["fingerprint"],
            "git_remote_refs_sha256": fingerprint["refs_sha256"],
            "git_config_sha256": fingerprint["config_sha256"],
            "git_old_oid": old_oid,
        }

    def fetch(
        self,
        pid: str,
        remote: str,
        expected_state_token: str,
        *,
        prune: bool = False,
        worktree_id: str = "main",
    ) -> GitOperationResult:
        selected_remote = self._validate_remote(remote)
        fingerprint = self._remote_preflight(
            pid,
            selected_remote,
            CapabilityRight.READ,
            worktree_id=worktree_id,
            effect_operation="fetch",
        )
        fingerprint_context = self._fingerprint_context(
            fingerprint,
            remote=selected_remote,
            use_push_url=False,
        )

        def dispatch(before: GitRepositoryState) -> tuple[str | None, dict[str, Any], Sequence[bytes]]:
            current_fingerprint = self.provider.remote_fingerprint(
                selected_remote,
                worktree=self._worktree_path(worktree_id),
            )
            if current_fingerprint["fingerprint"] != fingerprint["fingerprint"]:
                raise GitError(GitErrorCode.STALE_STATE.value, "Git remote configuration or refs changed before fetch", retryable=True)
            args = [
                "fetch",
                "--no-recurse-submodules",
                "--no-tags",
                "--no-prune-tags",
            ]
            if prune:
                args.append("--prune")
            else:
                args.append("--no-prune")
            args.append(selected_remote)
            self._mutation_command(
                before,
                "fetch",
                args,
                worktree_id=worktree_id,
                remote=selected_remote,
                expected_remote_fingerprint=fingerprint["fingerprint"],
            )
            return None, {
                "remote": selected_remote,
                "prune": prune,
                "remote_fingerprint": fingerprint["fingerprint"],
            }, ()

        return self._mutate(
            pid,
            "fetch",
            expected_state_token,
            dispatch,
            worktree_id=worktree_id,
            destructive=prune,
            extra={**fingerprint_context, "prune": prune},
            descriptor="primitive.git.fetch",
            additional_authorities=((self.remote_resource(selected_remote), CapabilityRight.READ, False),),
            remote=selected_remote,
        )

    def pull(
        self,
        pid: str,
        remote: str,
        expected_state_token: str,
        *,
        branch: str | None = None,
        strategy: str = "ff_only",
        worktree_id: str = "main",
    ) -> GitOperationResult:
        if strategy not in {"ff_only", "merge", "rebase"}:
            raise ValidationError("Git pull strategy must be ff_only, merge, or rebase")
        selected_remote = self._validate_remote(remote)
        selected_branch = self._pull_branch_for_authorization(
            pid,
            branch,
            expected_state_token=expected_state_token,
            worktree_id=worktree_id,
        )
        fingerprint = self._remote_preflight(
            pid,
            selected_remote,
            CapabilityRight.READ,
            worktree_id=worktree_id,
            effect_operation="fetch",
        )
        fingerprint_context = self._fingerprint_context(
            fingerprint,
            remote=selected_remote,
            use_push_url=False,
            remote_ref=f"refs/heads/{selected_branch}",
        )

        def dispatch(before: GitRepositoryState) -> tuple[str | None, dict[str, Any], Sequence[bytes]]:
            current_fingerprint = self.provider.remote_fingerprint(
                selected_remote,
                worktree=self._worktree_path(worktree_id),
            )
            if current_fingerprint["fingerprint"] != fingerprint["fingerprint"]:
                raise GitError(GitErrorCode.STALE_STATE.value, "Git remote configuration or refs changed before pull", retryable=True)
            branch_name = selected_branch
            tracking_ref = f"refs/remotes/{selected_remote}/{branch_name}"
            self._mutation_command(
                before,
                "pull",
                [
                    "fetch",
                    "--no-recurse-submodules",
                    "--no-tags",
                    "--no-prune",
                    "--no-prune-tags",
                    selected_remote,
                    f"+refs/heads/{branch_name}:{tracking_ref}",
                ],
                worktree_id=worktree_id,
                remote=selected_remote,
                expected_remote_fingerprint=fingerprint["fingerprint"],
            )
            fetched = self.provider.repository_state(worktree=self._worktree_path(worktree_id))
            target_oid = self._resolve_commit(tracking_ref, worktree_id=worktree_id)
            self._mark_git_source_carrier("commit", target_oid)
            if strategy == "ff_only":
                args = [
                    "merge",
                    "--ff-only",
                    "--no-edit",
                    target_oid,
                ]
            elif strategy == "merge":
                args = [
                    "merge",
                    "--no-edit",
                    "--no-gpg-sign",
                    "--commit",
                    "--no-squash",
                    target_oid,
                ]
            else:
                args = ["rebase", "--no-autostash", target_oid]
            try:
                self._mutation_command(fetched, "pull", args, worktree_id=worktree_id)
            except GitError as exc:
                raise GitError(
                    exc.code,
                    str(exc),
                    operation="pull",
                    retryable=exc.retryable,
                    details={**exc.details, "effect": "unknown", "fetch_completed": True},
                ) from exc
            current = self.provider.repository_state(worktree=self._worktree_path(worktree_id))
            return current.head_oid, {
                "remote": selected_remote,
                "branch": branch_name,
                "strategy": strategy,
                "remote_oid": target_oid,
                "old_head_oid": before.head_oid,
                "remote_fingerprint": fingerprint["fingerprint"],
            }, ()

        return self._mutate(
            pid,
            "pull",
            expected_state_token,
            dispatch,
            worktree_id=worktree_id,
            filesystem_rights=(
                CapabilityRight.READ,
                CapabilityRight.WRITE,
                CapabilityRight.DELETE,
            ),
            destructive=strategy == "rebase",
            extra={
                **fingerprint_context,
                "branch": selected_branch,
                "strategy": strategy,
            },
            descriptor="primitive.git.fetch",
            additional_authorities=((self.remote_resource(selected_remote), CapabilityRight.READ, False),),
            remote=selected_remote,
        )

    def _pull_branch_for_authorization(
        self,
        pid: str,
        branch: str | None,
        *,
        expected_state_token: str,
        worktree_id: str,
    ) -> str:
        if branch is not None:
            return self._validate_branch(branch)
        expected = self._validate_expected_token(expected_state_token)
        current = self.status(pid, worktree_id=worktree_id)
        if current.state.token != expected:
            raise GitError(
                GitErrorCode.STALE_STATE.value,
                "Git repository state changed after the preceding read",
                retryable=True,
                details={
                    "actual_state_token": current.state.token,
                    "effect_started": False,
                },
            )
        if current.branch is None:
            raise GitError(
                GitErrorCode.INVALID_REF.value,
                "detached HEAD requires an explicit pull branch",
            )
        return self._validate_branch(current.branch)

    def _canonical_local_ref(self, value: str) -> str:
        if value.startswith("refs/"):
            selected = self._validate_ref_name(value)
            if not selected.startswith(("refs/heads/", "refs/tags/")):
                raise GitError(GitErrorCode.INVALID_REF.value, "push local ref must be a branch or tag")
            return selected
        return f"refs/heads/{self._validate_branch(value)}"

    def _push_selection(
        self,
        *,
        remote: str,
        remote_ref: str,
        local_ref: str | None,
        delete: bool,
        force_with_lease_oid: str | None,
    ) -> tuple[str, str, str | None]:
        selected_remote = self._validate_remote(remote)
        selected_remote_ref = self._validate_ref_name(remote_ref)
        if not selected_remote_ref.startswith(("refs/heads/", "refs/tags/")):
            raise GitError(
                GitErrorCode.INVALID_REF.value,
                "push remote ref must be a branch or tag",
            )
        if delete:
            if local_ref is not None:
                raise ValidationError("remote ref deletion does not accept local_ref")
            selected_local_ref = None
        else:
            if local_ref is None:
                raise GitError(
                    GitErrorCode.INVALID_REF.value,
                    "push requires an explicit local_ref",
                )
            selected_local_ref = self._canonical_local_ref(local_ref)
        if (delete or force_with_lease_oid is not None) and force_with_lease_oid is None:
            raise GitError(
                GitErrorCode.INVALID_REF.value,
                "destructive push requires expected remote OID",
            )
        if force_with_lease_oid is not None and (
            not _OID_RE.fullmatch(force_with_lease_oid)
            or not any(character != "0" for character in force_with_lease_oid)
        ):
            raise GitError(
                GitErrorCode.INVALID_REF.value,
                "force-with-lease expected remote OID is invalid",
            )
        return selected_remote, selected_remote_ref, selected_local_ref

    def _push_flow_snapshot(
        self,
        *,
        expected_state_token: str,
        worktree_id: str,
        local_ref: str | None,
    ) -> _GitFlowSnapshot:
        if local_ref is None:
            empty_version = self._flow_state_version(())
            return _GitFlowSnapshot(
                context=DataFlowContext(),
                state_version=empty_version,
                state_version_resolver=lambda: empty_version,
            )
        return self._current_git_lineage_snapshot(
            expected_state_token,
            worktree_id=worktree_id,
            include_index=False,
            include_head=False,
            local_ref=local_ref,
        )

    def push(
        self,
        pid: str,
        remote: str,
        remote_ref: str,
        expected_state_token: str,
        *,
        local_ref: str | None = None,
        delete: bool = False,
        force_with_lease_oid: str | None = None,
        worktree_id: str = "main",
    ) -> GitOperationResult:
        selected_remote, selected_remote_ref, selected_local_ref = self._push_selection(
            remote=remote,
            remote_ref=remote_ref,
            local_ref=local_ref,
            delete=delete,
            force_with_lease_oid=force_with_lease_oid,
        )
        fingerprint = self._remote_preflight(
            pid,
            selected_remote,
            CapabilityRight.WRITE,
            worktree_id=worktree_id,
            effect_operation="push",
        )
        fingerprint_context = self._fingerprint_context(
            fingerprint,
            remote=selected_remote,
            use_push_url=True,
            remote_ref=selected_remote_ref,
            old_oid=force_with_lease_oid,
        )

        def read_push_flow(
            _scopes: Sequence[tuple[bytes | None, bool]],
        ) -> _GitFlowSnapshot:
            return self._push_flow_snapshot(
                expected_state_token=expected_state_token,
                worktree_id=worktree_id,
                local_ref=selected_local_ref,
            )

        def dispatch(before: GitRepositoryState) -> tuple[str | None, dict[str, Any], Sequence[bytes]]:
            if force_with_lease_oid is not None:
                expected_length = 40 if before.layout.object_format == "sha1" else 64
                if len(force_with_lease_oid) != expected_length:
                    raise GitError(
                        GitErrorCode.INVALID_REF.value,
                        "force-with-lease OID does not match repository object format",
                    )
            current_fingerprint = self.provider.remote_fingerprint(
                selected_remote,
                worktree=self._worktree_path(worktree_id),
            )
            if current_fingerprint["fingerprint"] != fingerprint["fingerprint"]:
                raise GitError(GitErrorCode.STALE_STATE.value, "Git remote configuration or refs changed before push", retryable=True)
            local_oid = (
                None
                if selected_local_ref is None
                else self._resolve_ref_oid(selected_local_ref, worktree_id=worktree_id)
            )
            args = [
                "push",
                "--porcelain",
                "--no-verify",
                "--no-follow-tags",
                "--signed=false",
            ]
            if force_with_lease_oid is not None:
                args.append(f"--force-with-lease={selected_remote_ref}:{force_with_lease_oid}")
            args.append(selected_remote)
            args.append(
                f":{selected_remote_ref}"
                if delete
                else f"{local_oid}:{selected_remote_ref}"
            )
            self._mutation_command(
                before,
                "push",
                args,
                worktree_id=worktree_id,
                remote=selected_remote,
                expected_remote_fingerprint=fingerprint["fingerprint"],
            )
            return local_oid, {
                "remote": selected_remote,
                "remote_ref": selected_remote_ref,
                "local_ref": selected_local_ref,
                "local_oid": local_oid,
                "delete": delete,
                "force_with_lease": force_with_lease_oid is not None,
                "expected_remote_oid": force_with_lease_oid,
                "remote_fingerprint": fingerprint["fingerprint"],
            }, ()

        return self._mutate(
            pid,
            "push",
            expected_state_token,
            dispatch,
            worktree_id=worktree_id,
            destructive=delete or force_with_lease_oid is not None,
            extra={
                **fingerprint_context,
                "local_ref": selected_local_ref,
                "remote_ref": selected_remote_ref,
                "delete": delete,
                "force_with_lease": force_with_lease_oid is not None,
            },
            descriptor="primitive.git.push",
            resource=self.remote_resource(selected_remote),
            primary_right=CapabilityRight.WRITE,
            additional_authorities=((self.repository_resource, CapabilityRight.READ, False),),
            remote=selected_remote,
            protected_flow_snapshot_resolver=read_push_flow,
        )

    @staticmethod
    def _canonical_pull_request_ref(value: str) -> str:
        if value.startswith("refs/"):
            selected = GitPrimitive._validate_ref_name(value)
            if not selected.startswith("refs/heads/"):
                raise GitError(GitErrorCode.INVALID_REF.value, "pull request refs must be local branches")
            return selected
        return f"refs/heads/{GitPrimitive._validate_branch(value)}"

    @staticmethod
    def _pull_request_snapshot_refs(pr_id: str) -> tuple[str, str]:
        if not _PR_ID_RE.fullmatch(pr_id):
            raise GitError(GitErrorCode.INVALID_REF.value, "invalid pull request id")
        prefix = f"refs/agent-libos/pull-requests/{pr_id}"
        return f"{prefix}/base", f"{prefix}/head"

    @staticmethod
    def _pull_request_payload(pull_request: GitPullRequest, *, review_bodies: dict[str, str] | None = None) -> bytes:
        bodies = dict(review_bodies or {})
        payload = {
            "schema": "agent-libos.git-pr.v1",
            "pr_id": pull_request.pr_id,
            "repository_id": pull_request.repository_id,
            "title": pull_request.title,
            "title_sha256": _sha256(pull_request.title.encode("utf-8")),
            "body": pull_request.body,
            "body_sha256": _sha256(pull_request.body.encode("utf-8")),
            "base_ref": pull_request.base_ref,
            "base_oid": pull_request.base_oid,
            "head_ref": pull_request.head_ref,
            "head_oid": pull_request.head_oid,
            "patch_sha256": pull_request.patch_sha256,
            "status": pull_request.status.value,
            "created_by": pull_request.created_by,
            "created_at": pull_request.created_at,
            "updated_at": pull_request.updated_at,
            "reviews": [
                {
                    "review_id": review.review_id,
                    "actor": review.actor,
                    "decision": review.decision.value,
                    "body": bodies.get(review.review_id, ""),
                    "body_sha256": review.body_sha256,
                    "created_at": review.created_at,
                }
                for review in pull_request.reviews
            ],
            "merged_oid": pull_request.merged_oid,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def _write_pull_request(
        self,
        pull_request: GitPullRequest,
        *,
        review_bodies: dict[str, str] | None = None,
        expected_sha256: str | None,
        create: bool = False,
        lineage_prebound: bool = False,
        capacity_preflighted: bool = False,
    ) -> GitPullRequest:
        data = self._pull_request_payload(
            pull_request,
            review_bodies=review_bodies,
        )
        if not capacity_preflighted:
            self._preflight_pull_request_metadata_capacity(
                pull_request.pr_id,
                data,
                create=create,
            )
        content_sha256 = _sha256(data)
        if not lineage_prebound:
            self._mark_git_output_carrier("pull_request", pull_request.pr_id)
            self._prebind_pull_request_lineage(pull_request.pr_id)
        self._mark_git_effect_started()
        persisted_sha256 = self.provider.write_pull_request_metadata(
            pull_request.pr_id,
            data,
            expected_sha256=expected_sha256,
            create=create,
        )
        if persisted_sha256 != content_sha256:
            raise GitError(
                GitErrorCode.UNKNOWN_EFFECT.value,
                "pull request metadata digest did not match persisted content",
                operation="pull_request_metadata",
            )
        return replace(
            pull_request,
            truncated=False,
            bytes=len(data),
            sha256=content_sha256,
        )

    def _preflight_pull_request_metadata_capacity(
        self,
        pr_id: str,
        data: bytes,
        *,
        create: bool,
    ) -> None:
        if len(data) > self.config.git.output_hard_limit_bytes:
            raise GitError(
                GitErrorCode.OUTPUT_TOO_LARGE.value,
                "pull request metadata exceeds its hard limit",
                operation="pull_request_metadata",
            )
        rows = self.provider.list_pull_request_metadata(
            limit=self.config.git.status_entry_hard_limit,
        )
        current = next(
            (current_data for current_id, current_data, _digest in rows if current_id == pr_id),
            None,
        )
        if create and current is not None:
            raise GitError(
                GitErrorCode.ALREADY_EXISTS.value,
                "pull request already exists",
                operation="pull_request_metadata",
            )
        if not create and current is None:
            return
        projected_count = len(rows) + (1 if create and current is None else 0)
        if projected_count > self.config.git.status_entry_hard_limit:
            raise GitError(
                GitErrorCode.OUTPUT_TOO_LARGE.value,
                "pull request metadata count exceeds its hard limit",
                operation="pull_request_metadata",
            )
        projected_bytes = sum(len(current_data) for _pr_id, current_data, _digest in rows)
        if current is not None:
            projected_bytes -= len(current)
        projected_bytes += len(data)
        if projected_bytes > self.config.git.output_hard_limit_bytes:
            raise GitError(
                GitErrorCode.OUTPUT_TOO_LARGE.value,
                "pull request metadata exceeds its aggregate limit",
                operation="pull_request_metadata",
            )

    @staticmethod
    def _parse_pull_request_metadata(data: bytes, *, expected_pr_id: str | None = None) -> tuple[GitPullRequest, dict[str, str]]:
        try:
            payload = json.loads(data.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GitError(GitErrorCode.UNSAFE_REPOSITORY.value, "pull request metadata is malformed") from exc
        if not isinstance(payload, dict) or payload.get("schema") != "agent-libos.git-pr.v1":
            raise GitError(GitErrorCode.UNSAFE_REPOSITORY.value, "pull request metadata schema is unsupported")
        try:
            pr_id = str(payload["pr_id"])
            if expected_pr_id is not None and pr_id != expected_pr_id:
                raise ValueError("pull request id mismatch")
            if not _PR_ID_RE.fullmatch(pr_id):
                raise ValueError("invalid pull request id")
            title = str(payload["title"])
            body = str(payload["body"])
            if _sha256(title.encode("utf-8")) != payload["title_sha256"]:
                raise ValueError("title digest mismatch")
            if _sha256(body.encode("utf-8")) != payload["body_sha256"]:
                raise ValueError("body digest mismatch")
            reviews: list[GitPullRequestReview] = []
            review_bodies: dict[str, str] = {}
            for raw_review in payload.get("reviews", []):
                review_body = str(raw_review.get("body", ""))
                body_sha256 = str(raw_review["body_sha256"])
                if _sha256(review_body.encode("utf-8")) != body_sha256:
                    raise ValueError("review digest mismatch")
                review = GitPullRequestReview(
                    review_id=str(raw_review["review_id"]),
                    actor=str(raw_review["actor"]),
                    decision=GitReviewDecision(str(raw_review["decision"])),
                    body_sha256=body_sha256,
                    created_at=str(raw_review["created_at"]),
                )
                reviews.append(review)
                review_bodies[review.review_id] = review_body
            pull_request = GitPullRequest(
                pr_id=pr_id,
                repository_id=str(payload["repository_id"]),
                title=title,
                body=body,
                base_ref=GitPrimitive._canonical_pull_request_ref(str(payload["base_ref"])),
                base_oid=str(payload["base_oid"]),
                head_ref=GitPrimitive._canonical_pull_request_ref(str(payload["head_ref"])),
                head_oid=str(payload["head_oid"]),
                patch_sha256=str(payload["patch_sha256"]),
                status=GitPullRequestStatus(str(payload["status"])),
                created_by=str(payload["created_by"]),
                created_at=str(payload["created_at"]),
                updated_at=str(payload["updated_at"]),
                reviews=reviews,
                merged_oid=str(payload["merged_oid"]) if payload.get("merged_oid") else None,
                truncated=False,
                bytes=len(data),
                sha256=_sha256(data),
            )
            if not _OID_RE.fullmatch(pull_request.base_oid) or not _OID_RE.fullmatch(pull_request.head_oid):
                raise ValueError("invalid pull request OID")
            if not re.fullmatch(r"[0-9a-f]{64}", pull_request.patch_sha256):
                raise ValueError("invalid patch digest")
            if pull_request.merged_oid is not None and not _OID_RE.fullmatch(pull_request.merged_oid):
                raise ValueError("invalid merged OID")
        except (KeyError, TypeError, ValueError) as exc:
            raise GitError(GitErrorCode.UNSAFE_REPOSITORY.value, "pull request metadata is malformed") from exc
        return pull_request, review_bodies

    def _read_pull_request(self, pr_id: str) -> tuple[GitPullRequest, dict[str, str], str]:
        if not _PR_ID_RE.fullmatch(pr_id):
            raise GitError(GitErrorCode.INVALID_REF.value, "invalid pull request id")
        item = self.provider.read_pull_request_metadata(pr_id)
        if item is None:
            raise GitError(GitErrorCode.NOT_FOUND.value, "pull request was not found")
        data, metadata_sha256 = item
        pull_request, bodies = self._parse_pull_request_metadata(data, expected_pr_id=pr_id)
        self._mark_git_source_carrier("pull_request", pr_id)
        return pull_request, bodies, metadata_sha256

    def _verify_pull_request_snapshot(self, pull_request: GitPullRequest) -> None:
        base_snapshot, head_snapshot = self._pull_request_snapshot_refs(pull_request.pr_id)
        base_oid = self._resolve_commit(base_snapshot, worktree_id="main")
        head_oid = self._resolve_commit(head_snapshot, worktree_id="main")
        if base_oid != pull_request.base_oid or head_oid != pull_request.head_oid:
            raise GitError(GitErrorCode.UNSAFE_REPOSITORY.value, "pull request snapshot refs do not match metadata")

    def create_pull_request(
        self,
        pid: str,
        title: str,
        body: str,
        base_ref: str,
        head_ref: str,
        expected_state_token: str,
    ) -> dict[str, Any]:
        if not isinstance(title, str) or not title.strip() or "\x00" in title or len(title.encode("utf-8")) > 4096:
            raise ValidationError("pull request title is invalid")
        if not isinstance(body, str) or "\x00" in body or len(body.encode("utf-8")) > 131_072:
            raise ValidationError("pull request body is invalid")
        selected_base = self._canonical_pull_request_ref(base_ref)
        selected_head = self._canonical_pull_request_ref(head_ref)
        selected_expected_state = self._validate_expected_token(expected_state_token)
        pr_id = self._stable_generated_id(
            "pr",
            pid,
            selected_expected_state,
            selected_base,
            selected_head,
            _sha256(title.encode("utf-8")),
            _sha256(body.encode("utf-8")),
        )
        if not _PR_ID_RE.fullmatch(pr_id):
            raise RuntimeError("generated pull request id is invalid")
        pr_resource = self.pull_request_resource(pr_id)
        created: list[GitPullRequest] = []

        def dispatch(before: GitRepositoryState) -> tuple[str | None, dict[str, Any], Sequence[bytes]]:
            base_oid = self._resolve_commit(selected_base, worktree_id="main")
            head_oid = self._resolve_commit(selected_head, worktree_id="main")
            patch_result = self._require_success(
                self._run(
                    [
                        "diff",
                        "--no-renames",
                        "--no-ext-diff",
                        "--no-textconv",
                        "--binary",
                        base_oid,
                        head_oid,
                        "--",
                    ],
                    worktree_id="main",
                    max_output_bytes=self.config.git.patch_hard_limit_bytes,
                ),
                "create_pull_request",
            )
            now = utc_now()
            pull_request = GitPullRequest(
                pr_id=pr_id,
                repository_id=before.layout.repository_id,
                title=title,
                body=body,
                base_ref=selected_base,
                base_oid=base_oid,
                head_ref=selected_head,
                head_oid=head_oid,
                patch_sha256=patch_result.stdout_sha256,
                status=GitPullRequestStatus.OPEN,
                created_by=pid,
                created_at=now,
                updated_at=now,
            )
            base_snapshot, head_snapshot = self._pull_request_snapshot_refs(pr_id)
            transaction = (
                f"create {base_snapshot} {base_oid}\n"
                f"create {head_snapshot} {head_oid}\n"
            ).encode("ascii")
            self._preflight_pull_request_metadata_capacity(
                pr_id,
                self._pull_request_payload(pull_request),
                create=True,
            )
            self._mark_git_output_carrier("pull_request", pr_id)
            self._prebind_pull_request_lineage(pr_id)
            self._mutation_command(
                before,
                "create_pull_request",
                ["update-ref", "--stdin"],
                worktree_id="main",
                stdin=transaction,
            )
            pull_request = self._write_pull_request(
                pull_request,
                expected_sha256=None,
                create=True,
                lineage_prebound=True,
                capacity_preflighted=True,
            )
            created.append(pull_request)
            return head_oid, {
                "pr_id": pr_id,
                "base_oid": base_oid,
                "head_oid": head_oid,
                "patch_sha256": patch_result.stdout_sha256,
                "title_sha256": _sha256(title.encode("utf-8")),
                "body_sha256": _sha256(body.encode("utf-8")),
                "patch_bytes": len(patch_result.stdout),
            }, ()

        operation_result = self._mutate(
            pid,
            "create_pull_request",
            selected_expected_state,
            dispatch,
            descriptor="primitive.git.pull_request",
            resource=pr_resource,
            primary_right=CapabilityRight.WRITE,
            additional_authorities=((self.repository_resource, CapabilityRight.WRITE, False),),
            additional_data_sinks=(DataSink(self.repository_resource),),
            extra={
                "pr_id": pr_id,
                "base_ref": selected_base,
                "head_ref": selected_head,
                "title_sha256": _sha256(title.encode("utf-8")),
                "body_sha256": _sha256(body.encode("utf-8")),
            },
            protected_flow_snapshot_resolver=lambda _scopes: self._current_range_flow_snapshot(
                selected_expected_state,
                base=selected_base,
                head=selected_head,
                worktree_id="main",
            ),
        )
        if len(created) != 1:
            raise GitError(GitErrorCode.UNKNOWN_EFFECT.value, "pull request creation outcome is unknown")
        pull_request = replace(created[0], state=operation_result.after)
        return {"pull_request": pull_request, "operation": operation_result}

    def list_pull_requests(
        self,
        pid: str,
        *,
        status: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        if limit <= 0 or limit > self.config.git.status_entry_hard_limit:
            raise ValidationError("pull request list limit is outside configured bounds")
        try:
            selected_status = (
                GitPullRequestStatus(status)
                if status is not None
                else None
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError("pull request status is invalid") from exc

        def read() -> tuple[dict[str, Any], dict[str, Any]]:
            before = self.provider.repository_state()
            scan_limit = self.config.git.status_entry_hard_limit
            rows = self.provider.list_pull_request_metadata(limit=scan_limit)
            pull_requests: list[GitPullRequest] = []
            bytes_read = 0
            digest = hashlib.sha256()
            matched_total = 0
            for pr_id, data, metadata_sha256 in rows:
                pull_request, _bodies = self._parse_pull_request_metadata(data, expected_pr_id=pr_id)
                self._verify_pull_request_snapshot(pull_request)
                if selected_status is not None and pull_request.status is not selected_status:
                    continue
                matched_total += 1
                bytes_read += len(data)
                digest.update(bytes.fromhex(metadata_sha256))
                if len(pull_requests) < limit:
                    pull_requests.append(pull_request)
            after = self.provider.repository_state()
            token = self._state_token(before)
            if token.token != self._state_token(after).token:
                raise GitError(GitErrorCode.STALE_STATE.value, "pull request list changed while being read", retryable=True)
            pull_requests = [replace(item, state=token) for item in pull_requests]
            payload = {
                "pull_requests": pull_requests,
                "truncated": matched_total > limit,
                "bytes": bytes_read,
                "sha256": digest.hexdigest(),
                "state": token,
            }
            return payload, {
                "pull_requests": len(pull_requests),
                "status": selected_status.value if selected_status is not None else None,
                "truncated": payload["truncated"],
                "bytes": bytes_read,
                "sha256": payload["sha256"],
                "state_token": token.token,
            }

        return self._read(
            pid,
            "list_pull_requests",
            read,
            resource="git_pr:workspace:*",
            extra={"status": selected_status.value if selected_status is not None else None, "limit": limit},
            flow_snapshot_resolver=lambda: self._git_read_lineage_snapshot(
                worktree_id="main",
                fixed_carriers=(("pull_request_collection", "workspace"),),
            ),
        )

    def inspect_pull_request(self, pid: str, pr_id: str) -> GitPullRequest:
        if not _PR_ID_RE.fullmatch(pr_id):
            raise GitError(GitErrorCode.INVALID_REF.value, "invalid pull request id")

        def read() -> tuple[GitPullRequest, dict[str, Any]]:
            before = self.provider.repository_state()
            pull_request, _bodies, metadata_sha256 = self._read_pull_request(pr_id)
            self._verify_pull_request_snapshot(pull_request)
            after = self.provider.repository_state()
            token = self._state_token(before)
            if token.token != self._state_token(after).token:
                raise GitError(GitErrorCode.STALE_STATE.value, "pull request changed while being read", retryable=True)
            result = replace(pull_request, state=token)
            return result, {
                "pr_id": pr_id,
                "status": result.status.value,
                "base_oid": result.base_oid,
                "head_oid": result.head_oid,
                "patch_sha256": result.patch_sha256,
                "metadata_sha256": metadata_sha256,
                "reviews": len(result.reviews),
                "state_token": token.token,
            }

        return self._read(
            pid,
            "inspect_pull_request",
            read,
            resource=self.pull_request_resource(pr_id),
            extra={"pr_id": pr_id},
            flow_snapshot_resolver=lambda: self._git_read_lineage_snapshot(
                worktree_id="main",
                fixed_carriers=(("pull_request", pr_id),),
            ),
        )

    def review_pull_request(
        self,
        pid: str,
        pr_id: str,
        decision: str,
        body: str,
        expected_state_token: str,
    ) -> dict[str, Any]:
        if not _PR_ID_RE.fullmatch(pr_id):
            raise GitError(GitErrorCode.INVALID_REF.value, "invalid pull request id")
        try:
            selected_decision = GitReviewDecision(decision)
        except (TypeError, ValueError) as exc:
            raise ValidationError("pull request review decision is invalid") from exc
        encoded_body = body.encode("utf-8")
        if b"\x00" in encoded_body or len(encoded_body) > 131_072:
            raise ValidationError("pull request review body is invalid")
        updated: list[GitPullRequest] = []

        def dispatch(before: GitRepositoryState) -> tuple[str | None, dict[str, Any], Sequence[bytes]]:
            pull_request, bodies, metadata_sha256 = self._read_pull_request(pr_id)
            self._verify_pull_request_snapshot(pull_request)
            if pull_request.status is not GitPullRequestStatus.OPEN:
                raise GitError(GitErrorCode.CONFLICT.value, "only open pull requests can be reviewed")
            review = GitPullRequestReview(
                review_id=new_id("review"),
                actor=pid,
                decision=selected_decision,
                body_sha256=_sha256(encoded_body),
                created_at=utc_now(),
            )
            bodies[review.review_id] = body
            current = replace(
                pull_request,
                reviews=[*pull_request.reviews, review],
                updated_at=utc_now(),
            )
            current = self._write_pull_request(
                current,
                review_bodies=bodies,
                expected_sha256=metadata_sha256,
            )
            updated.append(current)
            return None, {
                "pr_id": pr_id,
                "review_id": review.review_id,
                "decision": selected_decision.value,
                "body_sha256": review.body_sha256,
                "body_bytes": len(encoded_body),
            }, ()

        primary_right = (
            CapabilityRight.WRITE
            if selected_decision is GitReviewDecision.COMMENT
            else CapabilityRight.APPROVE
        )
        operation_result = self._mutate(
            pid,
            "review_pull_request",
            expected_state_token,
            dispatch,
            descriptor="primitive.git.pull_request",
            resource=self.pull_request_resource(pr_id),
            primary_right=primary_right,
            extra={
                "pr_id": pr_id,
                "decision": selected_decision.value,
                "body_sha256": _sha256(encoded_body),
                "body_bytes": len(encoded_body),
            },
            protected_flow_snapshot_resolver=lambda _scopes: self._current_git_read_flow_snapshot(
                expected_state_token,
                worktree_id="main",
                fixed_carriers=(("pull_request", pr_id),),
            ),
        )
        if len(updated) != 1:
            raise GitError(GitErrorCode.UNKNOWN_EFFECT.value, "pull request review outcome is unknown")
        return {"pull_request": replace(updated[0], state=operation_result.after), "operation": operation_result}

    def close_pull_request(
        self,
        pid: str,
        pr_id: str,
        expected_state_token: str,
    ) -> dict[str, Any]:
        if not _PR_ID_RE.fullmatch(pr_id):
            raise GitError(GitErrorCode.INVALID_REF.value, "invalid pull request id")
        updated: list[GitPullRequest] = []

        def dispatch(before: GitRepositoryState) -> tuple[str | None, dict[str, Any], Sequence[bytes]]:
            pull_request, bodies, metadata_sha256 = self._read_pull_request(pr_id)
            self._verify_pull_request_snapshot(pull_request)
            if pull_request.status is not GitPullRequestStatus.OPEN:
                raise GitError(GitErrorCode.CONFLICT.value, "only open pull requests can be closed")
            current = replace(pull_request, status=GitPullRequestStatus.CLOSED, updated_at=utc_now())
            current = self._write_pull_request(
                current,
                review_bodies=bodies,
                expected_sha256=metadata_sha256,
            )
            updated.append(current)
            return None, {"pr_id": pr_id, "status": current.status.value}, ()

        operation_result = self._mutate(
            pid,
            "close_pull_request",
            expected_state_token,
            dispatch,
            descriptor="primitive.git.pull_request",
            resource=self.pull_request_resource(pr_id),
            primary_right=CapabilityRight.DELETE,
            extra={"pr_id": pr_id},
            protected_flow_snapshot_resolver=lambda _scopes: self._current_git_read_flow_snapshot(
                expected_state_token,
                worktree_id="main",
                fixed_carriers=(("pull_request", pr_id),),
            ),
        )
        if len(updated) != 1:
            raise GitError(GitErrorCode.UNKNOWN_EFFECT.value, "pull request close outcome is unknown")
        return {"pull_request": replace(updated[0], state=operation_result.after), "operation": operation_result}

    def merge_pull_request(
        self,
        pid: str,
        pr_id: str,
        expected_state_token: str,
        *,
        strategy: str = "fast_forward",
        worktree_id: str = "main",
    ) -> dict[str, Any]:
        if not _PR_ID_RE.fullmatch(pr_id):
            raise GitError(GitErrorCode.INVALID_REF.value, "invalid pull request id")
        if strategy not in {"fast_forward", "merge", "squash"}:
            raise ValidationError("pull request merge strategy must be fast_forward, merge, or squash")
        updated: list[GitPullRequest] = []

        def dispatch(before: GitRepositoryState) -> tuple[str | None, dict[str, Any], Sequence[bytes]]:
            pull_request, bodies, metadata_sha256 = self._read_pull_request(pr_id)
            self._verify_pull_request_snapshot(pull_request)
            if pull_request.status is not GitPullRequestStatus.OPEN:
                raise GitError(GitErrorCode.CONFLICT.value, "only open pull requests can be merged")
            base_current = self._resolve_commit(pull_request.base_ref, worktree_id=worktree_id)
            head_current = self._resolve_commit(pull_request.head_ref, worktree_id=worktree_id)
            if base_current != pull_request.base_oid or head_current != pull_request.head_oid:
                raise GitError(GitErrorCode.STALE_STATE.value, "pull request base or head advanced before merge", retryable=True)
            if before.head_ref != pull_request.base_ref or before.head_oid != pull_request.base_oid:
                raise GitError(GitErrorCode.STALE_STATE.value, "pull request base must be checked out at its recorded OID", retryable=True)
            parsed_status = self._parse_status(before, limit=self.config.git.status_entry_hard_limit)
            if parsed_status.entries:
                raise GitError(GitErrorCode.DIRTY_WORKTREE.value, "pull request merge requires a clean worktree")
            self._mark_git_source_carrier("commit", pull_request.head_oid)
            if strategy == "fast_forward":
                args = ["merge", "--ff-only", "--no-edit", pull_request.head_oid]
                self._mutation_command(before, "merge_pull_request", args, worktree_id=worktree_id)
            elif strategy == "merge":
                args = ["merge", "--no-ff", "--no-edit", "--no-gpg-sign", pull_request.head_oid]
                self._mutation_command(before, "merge_pull_request", args, worktree_id=worktree_id)
            else:
                self._mutation_command(
                    before,
                    "merge_pull_request",
                    ["merge", "--squash", pull_request.head_oid],
                    worktree_id=worktree_id,
                )
                staged = self.provider.repository_state(worktree=self._worktree_path(worktree_id))
                identity = self._run(["var", "GIT_AUTHOR_IDENT"], worktree_id=worktree_id, max_output_bytes=65536)
                if identity.returncode != 0:
                    raise GitError(GitErrorCode.IDENTITY_MISSING.value, "Git commit identity is missing")
                message = f"Merge pull request {pr_id} (squash)".encode("utf-8")
                self._mutation_command(
                    staged,
                    "merge_pull_request",
                    ["commit", "--no-verify", "--no-gpg-sign", "-F", "-"],
                    worktree_id=worktree_id,
                    stdin=message,
                )
            merged_state = self.provider.repository_state(worktree=self._worktree_path(worktree_id))
            if merged_state.head_oid is None:
                raise GitError(GitErrorCode.UNKNOWN_EFFECT.value, "pull request merge produced no HEAD")
            current = replace(
                pull_request,
                status=GitPullRequestStatus.MERGED,
                merged_oid=merged_state.head_oid,
                updated_at=utc_now(),
            )
            current = self._write_pull_request(
                current,
                review_bodies=bodies,
                expected_sha256=metadata_sha256,
            )
            updated.append(current)
            return merged_state.head_oid, {
                "pr_id": pr_id,
                "strategy": strategy,
                "base_oid": pull_request.base_oid,
                "head_oid": pull_request.head_oid,
                "merged_oid": merged_state.head_oid,
                "patch_sha256": pull_request.patch_sha256,
            }, ()

        operation_result = self._mutate(
            pid,
            "merge_pull_request",
            expected_state_token,
            dispatch,
            worktree_id=worktree_id,
            filesystem_rights=(
                CapabilityRight.READ,
                CapabilityRight.WRITE,
                CapabilityRight.DELETE,
            ),
            descriptor="primitive.git.pull_request",
            resource=self.pull_request_resource(pr_id),
            primary_right=CapabilityRight.APPROVE,
            mandatory_primary_approval=True,
            additional_authorities=(
                (self.repository_resource, CapabilityRight.WRITE, False),
                (self.repository_resource, CapabilityRight.DELETE, False),
                (self.repository_resource, CapabilityRight.ADMIN, False),
            ),
            additional_data_sinks=(DataSink(self.repository_resource),),
            extra={"pr_id": pr_id, "strategy": strategy},
            protected_flow_snapshot_resolver=lambda _scopes: self._current_git_read_flow_snapshot(
                expected_state_token,
                worktree_id=worktree_id,
                fixed_carriers=(("pull_request", pr_id),),
            ),
        )
        if len(updated) != 1:
            raise GitError(GitErrorCode.UNKNOWN_EFFECT.value, "pull request merge outcome is unknown")
        return {"pull_request": replace(updated[0], state=operation_result.after), "operation": operation_result}

    async def _async_call(self, function: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
        """Run a synchronous Git boundary without blocking the active loop."""

        if self.data_flow is None:
            raise ValidationError("Git data-flow manager is not attached")
        return await self.data_flow.run_sync_in_worker(function, *args, **kwargs)

    async def arepository_info(self, pid: str, *, worktree_id: str = "main") -> GitRepositoryInfo:
        return await self._async_call(self.repository_info, pid, worktree_id=worktree_id)

    async def astatus(
        self,
        pid: str,
        *,
        worktree_id: str = "main",
        limit: int | None = None,
    ) -> GitStatusResult:
        return await self._async_call(self.status, pid, worktree_id=worktree_id, limit=limit)

    async def adiff(
        self,
        pid: str,
        *,
        scope: str = "worktree",
        base: str | None = None,
        head: str | None = None,
        paths: Iterable[str | GitPath | dict[str, Any]] = (),
        worktree_id: str = "main",
        max_bytes: int | None = None,
    ) -> GitDiffResult:
        return await self._async_call(
            self.diff,
            pid,
            scope=scope,
            base=base,
            head=head,
            paths=paths,
            worktree_id=worktree_id,
            max_bytes=max_bytes,
        )

    async def alog(
        self,
        pid: str,
        *,
        ref: str | None = None,
        limit: int | None = None,
        worktree_id: str = "main",
    ) -> dict[str, Any]:
        return await self._async_call(self.log, pid, ref=ref, limit=limit, worktree_id=worktree_id)

    async def ashow(
        self,
        pid: str,
        ref: str,
        *,
        worktree_id: str = "main",
        max_bytes: int | None = None,
    ) -> dict[str, Any]:
        return await self._async_call(
            self.show,
            pid,
            ref,
            worktree_id=worktree_id,
            max_bytes=max_bytes,
        )

    async def ablame(
        self,
        pid: str,
        path: str | GitPath | dict[str, Any],
        *,
        ref: str | None = None,
        worktree_id: str = "main",
        max_bytes: int | None = None,
    ) -> dict[str, Any]:
        return await self._async_call(
            self.blame,
            pid,
            path,
            ref=ref,
            worktree_id=worktree_id,
            max_bytes=max_bytes,
        )

    async def alist_refs(
        self,
        pid: str,
        *,
        kind: str = "all",
        limit: int = 200,
        worktree_id: str = "main",
    ) -> dict[str, Any]:
        return await self._async_call(
            self.list_refs,
            pid,
            kind=kind,
            limit=limit,
            worktree_id=worktree_id,
        )

    async def alist_remotes(self, pid: str, *, worktree_id: str = "main") -> dict[str, Any]:
        return await self._async_call(self.list_remotes, pid, worktree_id=worktree_id)

    async def alist_worktrees(self, pid: str) -> dict[str, Any]:
        return await self._async_call(self.list_worktrees, pid)

    async def astage(
        self,
        pid: str,
        paths: Iterable[str | GitPath | dict[str, Any]],
        expected_state_token: str,
        *,
        worktree_id: str = "main",
    ) -> GitOperationResult:
        return await self._async_call(
            self.stage,
            pid,
            paths,
            expected_state_token,
            worktree_id=worktree_id,
        )

    async def aunstage(
        self,
        pid: str,
        paths: Iterable[str | GitPath | dict[str, Any]],
        expected_state_token: str,
        *,
        worktree_id: str = "main",
    ) -> GitOperationResult:
        return await self._async_call(
            self.unstage,
            pid,
            paths,
            expected_state_token,
            worktree_id=worktree_id,
        )

    async def acommit(
        self,
        pid: str,
        message: str,
        expected_state_token: str,
        *,
        amend: bool = False,
        worktree_id: str = "main",
    ) -> GitOperationResult:
        return await self._async_call(
            self.commit,
            pid,
            message,
            expected_state_token,
            amend=amend,
            worktree_id=worktree_id,
        )

    async def arestore(
        self,
        pid: str,
        paths: Iterable[str | GitPath | dict[str, Any]],
        expected_state_token: str,
        *,
        staged: bool = False,
        worktree: bool = True,
        source: str | None = None,
        worktree_id: str = "main",
    ) -> GitOperationResult:
        return await self._async_call(
            self.restore,
            pid,
            paths,
            expected_state_token,
            staged=staged,
            worktree=worktree,
            source=source,
            worktree_id=worktree_id,
        )

    async def abranch(
        self,
        pid: str,
        action: str,
        name: str,
        expected_state_token: str,
        *,
        start: str | None = None,
        new_name: str | None = None,
        force: bool = False,
        worktree_id: str = "main",
    ) -> GitOperationResult:
        return await self._async_call(
            self.branch,
            pid,
            action,
            name,
            expected_state_token,
            start=start,
            new_name=new_name,
            force=force,
            worktree_id=worktree_id,
        )

    async def aswitch(
        self,
        pid: str,
        target: str,
        expected_state_token: str,
        *,
        create: bool = False,
        start: str | None = None,
        detach: bool = False,
        force: bool = False,
        worktree_id: str = "main",
    ) -> GitOperationResult:
        return await self._async_call(
            self.switch,
            pid,
            target,
            expected_state_token,
            create=create,
            start=start,
            detach=detach,
            force=force,
            worktree_id=worktree_id,
        )

    async def atag(
        self,
        pid: str,
        action: str,
        name: str,
        expected_state_token: str,
        *,
        target: str | None = None,
        message: str | None = None,
        force: bool = False,
        worktree_id: str = "main",
    ) -> GitOperationResult:
        return await self._async_call(
            self.tag,
            pid,
            action,
            name,
            expected_state_token,
            target=target,
            message=message,
            force=force,
            worktree_id=worktree_id,
        )

    async def aintegrate(
        self,
        pid: str,
        operation: str,
        expected_state_token: str,
        *,
        ref: str | None = None,
        abort_kind: str | None = None,
        worktree_id: str = "main",
    ) -> GitOperationResult:
        return await self._async_call(
            self.integrate,
            pid,
            operation,
            expected_state_token,
            ref=ref,
            abort_kind=abort_kind,
            worktree_id=worktree_id,
        )

    async def astash(
        self,
        pid: str,
        action: str,
        expected_state_token: str,
        *,
        index: int = 0,
        include_untracked: bool = False,
        reinstate_index: bool = False,
        worktree_id: str = "main",
    ) -> GitOperationResult:
        return await self._async_call(
            self.stash,
            pid,
            action,
            expected_state_token,
            index=index,
            include_untracked=include_untracked,
            reinstate_index=reinstate_index,
            worktree_id=worktree_id,
        )

    async def areset(
        self,
        pid: str,
        target: str,
        expected_state_token: str,
        *,
        mode: str = "mixed",
        worktree_id: str = "main",
    ) -> GitOperationResult:
        return await self._async_call(
            self.reset,
            pid,
            target,
            expected_state_token,
            mode=mode,
            worktree_id=worktree_id,
        )

    async def aclean(
        self,
        pid: str,
        expected_state_token: str,
        *,
        paths: Iterable[str | GitPath | dict[str, Any]] = (),
        directories: bool = False,
        ignored: bool = False,
        worktree_id: str = "main",
    ) -> GitOperationResult:
        return await self._async_call(
            self.clean,
            pid,
            expected_state_token,
            paths=paths,
            directories=directories,
            ignored=ignored,
            worktree_id=worktree_id,
        )

    async def aworktree(
        self,
        pid: str,
        action: str,
        expected_state_token: str,
        *,
        ref: str | None = None,
        new_branch: str | None = None,
        managed_worktree_id: str | None = None,
    ) -> GitOperationResult:
        return await self._async_call(
            self.worktree,
            pid,
            action,
            expected_state_token,
            ref=ref,
            new_branch=new_branch,
            managed_worktree_id=managed_worktree_id,
        )

    async def acreate_patch(
        self,
        pid: str,
        *,
        scope: str = "worktree",
        base: str | None = None,
        head: str | None = None,
        paths: Iterable[str | GitPath | dict[str, Any]] = (),
        worktree_id: str = "main",
    ) -> GitPatchArtifact:
        return await self._async_call(
            self.create_patch,
            pid,
            scope=scope,
            base=base,
            head=head,
            paths=paths,
            worktree_id=worktree_id,
        )

    async def aapply_patch(
        self,
        pid: str,
        patch_oid: str,
        expected_state_token: str,
        *,
        index: bool = False,
        worktree_id: str = "main",
    ) -> GitOperationResult:
        return await self._async_call(
            self.apply_patch,
            pid,
            patch_oid,
            expected_state_token,
            index=index,
            worktree_id=worktree_id,
        )

    async def afetch(
        self,
        pid: str,
        remote: str,
        expected_state_token: str,
        *,
        prune: bool = False,
        worktree_id: str = "main",
    ) -> GitOperationResult:
        return await self._async_call(
            self.fetch,
            pid,
            remote,
            expected_state_token,
            prune=prune,
            worktree_id=worktree_id,
        )

    async def apull(
        self,
        pid: str,
        remote: str,
        expected_state_token: str,
        *,
        branch: str | None = None,
        strategy: str = "ff_only",
        worktree_id: str = "main",
    ) -> GitOperationResult:
        return await self._async_call(
            self.pull,
            pid,
            remote,
            expected_state_token,
            branch=branch,
            strategy=strategy,
            worktree_id=worktree_id,
        )

    async def apush(
        self,
        pid: str,
        remote: str,
        remote_ref: str,
        expected_state_token: str,
        *,
        local_ref: str | None = None,
        delete: bool = False,
        force_with_lease_oid: str | None = None,
        worktree_id: str = "main",
    ) -> GitOperationResult:
        return await self._async_call(
            self.push,
            pid,
            remote,
            remote_ref,
            expected_state_token,
            local_ref=local_ref,
            delete=delete,
            force_with_lease_oid=force_with_lease_oid,
            worktree_id=worktree_id,
        )

    async def acreate_pull_request(
        self,
        pid: str,
        title: str,
        body: str,
        base_ref: str,
        head_ref: str,
        expected_state_token: str,
    ) -> dict[str, Any]:
        return await self._async_call(
            self.create_pull_request,
            pid,
            title,
            body,
            base_ref,
            head_ref,
            expected_state_token,
        )

    async def alist_pull_requests(
        self,
        pid: str,
        *,
        status: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        return await self._async_call(self.list_pull_requests, pid, status=status, limit=limit)

    async def ainspect_pull_request(self, pid: str, pr_id: str) -> GitPullRequest:
        return await self._async_call(self.inspect_pull_request, pid, pr_id)

    async def areview_pull_request(
        self,
        pid: str,
        pr_id: str,
        decision: str,
        body: str,
        expected_state_token: str,
    ) -> dict[str, Any]:
        return await self._async_call(
            self.review_pull_request,
            pid,
            pr_id,
            decision,
            body,
            expected_state_token,
        )

    async def aclose_pull_request(
        self,
        pid: str,
        pr_id: str,
        expected_state_token: str,
    ) -> dict[str, Any]:
        return await self._async_call(self.close_pull_request, pid, pr_id, expected_state_token)

    async def amerge_pull_request(
        self,
        pid: str,
        pr_id: str,
        expected_state_token: str,
        *,
        strategy: str = "fast_forward",
        worktree_id: str = "main",
    ) -> dict[str, Any]:
        return await self._async_call(
            self.merge_pull_request,
            pid,
            pr_id,
            expected_state_token,
            strategy=strategy,
            worktree_id=worktree_id,
        )
