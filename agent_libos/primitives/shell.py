from __future__ import annotations

import os
import hashlib
import inspect
import math
import subprocess
import shutil
from dataclasses import dataclass, replace
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Protocol
from urllib.parse import urlsplit

from agent_libos.capability.manager import CapabilityManager
from agent_libos.capability.rules import AUTHORITY_RULES_KEY, RuleMatch, ShellRuleEngine
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig, ShellPolicyLevel
from agent_libos.models import (
    AuthorityRisk,
    AuthorityRule,
    Capability,
    CapabilityDecision,
    CapabilityEffect,
    CapabilityRight,
    DataSink,
    EventType,
    ResourceUsage,
    SandboxProfile,
)
from agent_libos.models.exceptions import CapabilityDenied, HumanApprovalRequired, ResourceLimitExceeded, ValidationError
from agent_libos.primitives.git_command_policy import (
    harden_read_only_git_argv,
    validate_and_normalize_raw_git,
)
from agent_libos.ports import AuditPort, EventPort
from agent_libos.substrate import (
    CommandMetrics,
    CommandResult,
    ExecutableSnapshot,
    executable_content_sha256,
    LocalShellProvider,
    resolve_runtime_python_alias,
    ShellProvider,
    snapshot_executable,
    SubprocessLimitExceeded,
    SubprocessLimits,
    SubprocessTimeoutExpired,
)
from agent_libos.sdk import (
    ProtectedOperationEvidence,
    ProtectedOperationInvocation,
    ProtectedOperationSDK,
    ProviderPhase,
    ResourceSettlement,
)

_TOOL_DEFAULTS = DEFAULT_CONFIG.tools

_WINDOWS_EXECUTABLE_SUFFIXES = (".exe", ".cmd", ".bat", ".com", ".ps1")


@dataclass(frozen=True)
class ShellPolicyDecision:
    allowed: bool
    ask_human: bool
    reason: str
    policy_level: str | None
    matched_rule: tuple[str, ...] | None = None
    high_risk: bool = False
    consume_once: bool = False
    consume_capability_id: str | None = None
    risk: AuthorityRisk = AuthorityRisk.MEDIUM
    rule_id: str | None = None
    rule_effect: CapabilityEffect = CapabilityEffect.ASK
    sandbox_profile: SandboxProfile | None = None
    authority_decision: CapabilityDecision | None = None


class ShellExecutionPolicy(Protocol):
    """Shared launch-policy contract implemented by ``ShellAdapter``.

    Interactive and one-shot process adapters need the same argv validation,
    authority, data-flow Sink, and provider-identity rules.  The protocol keeps
    Runtime Modules on a narrow public surface without adding a forwarding
    object or exposing ``ShellAdapter`` private implementation details.
    """

    def validate_argv(self, argv: list[str]) -> list[str]:
        ...

    def harden_read_only_git_argv(self, argv: list[str]) -> list[str]:
        ...

    def resource_for(self, argv: list[str]) -> str:
        ...

    def enforce_workspace_argv_scope(self, argv: list[str], *, cwd: str) -> None:
        ...

    def authorize_operation(
        self,
        pid: str,
        argv: list[str],
        resource: str,
        *,
        timeout: float,
        cwd: str,
        adapter: str,
        primitive: str,
        operation: str,
        authority_operation: str,
        include_timeout_in_authority: bool,
        continuous_session: bool = False,
        extra_context: dict[str, Any] | None = None,
    ) -> ShellPolicyDecision:
        ...

    def executable_data_sink(
        self,
        namespace: str,
        argv0: str,
        *,
        cwd: str | os.PathLike[str] | None = None,
    ) -> DataSink:
        ...

    def subprocess_limits(self, pid: str) -> SubprocessLimits | None:
        ...

    def operation_context(
        self,
        pid: str,
        argv: list[str],
        resource: str,
        *,
        timeout: float,
        cwd: str,
        profile: SandboxProfile,
        adapter: str = "shell",
        primitive: str = "runtime.shell.run",
        operation: str = "shell.run",
        authority_operation: str = "shell.run",
        include_timeout: bool = True,
        continuous_session: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...

    def resolve_provider_argv(
        self,
        argv: list[str],
        *,
        cwd: str | os.PathLike[str] | None,
        provider: Any | None = None,
    ) -> list[str]:
        ...

    def require_provider_executable_identity(
        self,
        argv0: str,
        *,
        expected: str,
        cwd: str | os.PathLike[str] | None,
    ) -> None:
        ...

    def snapshot_executable_for_dispatch(
        self,
        *,
        pid: str,
        provider: Any,
        requested_argv0: str,
        provider_argv0: str,
        cwd: str | os.PathLike[str] | None,
        expected_sink: DataSink,
        expected_executable_identity: str,
        flow_context: Any,
        data_flow_payload: Any,
    ) -> ExecutableSnapshot | None:
        ...

    def profile_json(
        self,
        profile: SandboxProfile | None,
    ) -> dict[str, Any] | None:
        ...

    def approval_constraints(
        self,
        argv: list[str],
        decision: ShellPolicyDecision,
        *,
        timeout: float,
        cwd: str,
        operation: str = "shell.run",
        include_timeout: bool = True,
        extra_conditions: dict[str, Any] | None = None,
        description: str = "one-shot human approval for exact shell argv",
    ) -> dict[str, Any]:
        ...


class ShellAdapter:
    """Capability-checked shell primitive.

    Commands are accepted only as argv arrays and are executed by the substrate
    with shell=False. Allow/ask decisions use exact token rules; no glob,
    substring, or shell-style parsing is used for whitelist matching.
    """

    def __init__(
        self,
        capabilities: CapabilityManager,
        audit: AuditPort,
        events: EventPort | None = None,
        *,
        protected_operations: ProtectedOperationSDK,
        cwd: str | os.PathLike[str] | None = None,
        human: Any | None = None,
        provider: ShellProvider | None = None,
        config: AgentLibOSConfig | None = None,
        resources: Any | None = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.capabilities = capabilities
        self.audit = audit
        self.events = events
        self.protected_operations = protected_operations
        self.human = human
        self.resources = resources
        self.provider = provider or LocalShellProvider(cwd or ".")
        provider_cwd = getattr(self.provider, "cwd", None)
        self.cwd = Path(cwd if cwd is not None else provider_cwd or ".").resolve()
        self.rule_engine = ShellRuleEngine(
            self._configured_rules(),
            evaluator=self.capabilities.evaluator,
        )

    def run(
        self,
        pid: str,
        argv: list[str],
        timeout: float = _TOOL_DEFAULTS.shell_timeout_s,
        cwd: str | os.PathLike[str] | None = None,
        *,
        source_oids: list[str] | tuple[str, ...] | None = None,
    ) -> CommandResult:
        checked = self.validate_argv(argv)
        selected_timeout = self._validate_timeout(timeout)
        resource = self.resource_for(checked)
        selected_cwd = os.fspath(cwd) if cwd is not None else "."
        self.enforce_workspace_argv_scope(checked, cwd=selected_cwd)
        decision = self._authorize(pid, checked, resource, timeout=selected_timeout, cwd=selected_cwd)
        if not decision.allowed and not decision.ask_human:
            raise CapabilityDenied(f"{pid} denied shell execute on {resource}: {decision.reason}")
        dispatch_argv = self.harden_read_only_git_argv(checked)
        sink = self.executable_data_sink("shell", dispatch_argv[0], cwd=cwd)
        executable_identity = sink.identity.split(":", 1)[1]
        flow_context = self._data_flow().context_from_source_oids(pid, source_oids)
        data_flow_payload = {
            "argv": [executable_identity, *dispatch_argv[1:]],
            "cwd": os.fspath(cwd) if cwd is not None else None,
        }
        self._data_flow().authorize_egress(
            pid=pid,
            sink=sink,
            context=flow_context,
            payload=data_flow_payload,
            operation="shell.run",
        )
        if decision.ask_human:
            self._request_human_approval(
                pid,
                checked,
                resource,
                decision,
                timeout=selected_timeout,
                cwd=cwd,
                source_oids=source_oids,
            )
        if not decision.allowed:
            raise CapabilityDenied(f"{pid} denied shell execute on {resource}: {decision.reason}")
        provider_argv = list(dispatch_argv)
        effect_context = {
            "argv": list(checked),
            "provider_argv": list(data_flow_payload["argv"]),
            "resource": resource,
            "timeout_s": selected_timeout,
            "cwd": os.fspath(cwd) if cwd is not None else None,
            "policy_level": decision.policy_level,
            "high_risk": decision.high_risk,
            "risk": decision.risk.value,
            "rule_id": decision.rule_id,
            "rule_effect": decision.rule_effect.value,
            "sandbox_profile": self.profile_json(decision.sandbox_profile),
        }
        limits = self.subprocess_limits(pid)
        provider_kwargs = self._provider_run_kwargs(timeout=selected_timeout, cwd=cwd, limits=limits)
        intent_record = self._record_run_intent(
            pid,
            resource,
            checked,
            decision,
            timeout=selected_timeout,
            cwd=cwd,
        )
        correlation_id = intent_record.record_id
        capability_decision = decision.authority_decision
        if capability_decision is None or not capability_decision.allowed:
            raise CapabilityDenied(
                "allowed shell policy decision is missing its capability authority"
            )

        def revalidate_shell_authority() -> tuple[CapabilityDecision, ...]:
            current = self._authorize(
                pid,
                checked,
                resource,
                timeout=selected_timeout,
                cwd=selected_cwd,
            )
            authority = current.authority_decision
            if not current.allowed or current.ask_human or authority is None:
                raise CapabilityDenied(
                    f"{pid} shell authority changed before provider dispatch: {current.reason}"
                )
            return (authority,)

        def revalidate_shell_sink() -> DataSink:
            return self.executable_data_sink("shell", provider_argv[0], cwd=cwd)

        invocation = ProtectedOperationInvocation(
            pid=pid,
            actor=pid,
            target=resource,
            decisions=(capability_decision,),
            canonical_args=self.operation_context(
                pid,
                checked,
                resource,
                timeout=selected_timeout,
                cwd=selected_cwd,
                profile=decision.sandbox_profile,
            ),
            observation={
                **effect_context,
                "argv": list(checked),
                "provider_argv": list(provider_argv),
            },
            authority_revalidator=revalidate_shell_authority,
            restore_not_started=lambda: self.audit.record(
                actor=pid,
                action="primitive.shell.failed",
                target=resource,
                decision={"effect_outcome": "not_started", "provider_started": False},
                correlation_id=correlation_id,
                parent_record_id=intent_record.record_id,
            ),
            failure_evidence=lambda error, phase: self._protected_failure_evidence(
                pid,
                resource,
                checked,
                cwd,
                intent_record,
                error,
                phase,
            ),
            data_sink=sink,
            data_sink_revalidator=revalidate_shell_sink,
            data_flow_context=flow_context,
            data_flow_ingress_context=self._data_flow().unclassified_ingress_context(
                flow_context,
                origin="external:shell",
            ),
            data_flow_payload=data_flow_payload,
            data_flow_operation="shell.run",
        )
        with self._protected().start("primitive.shell.run", invocation, provider=self.provider) as operation:
            resolver = getattr(self.provider, "resolve_argv", None)
            if callable(resolver):
                provider_argv = operation.call(
                    ProviderPhase("resolve_argv", information_flow=True),
                    self.resolve_provider_argv,
                    dispatch_argv,
                    cwd=cwd,
                )
            self.require_provider_executable_identity(
                provider_argv[0],
                expected=executable_identity,
                cwd=cwd,
            )
            effect_context["provider_argv"] = list(provider_argv)
            executable_snapshot = self.snapshot_executable_for_dispatch(
                pid=pid,
                provider=self.provider,
                requested_argv0=dispatch_argv[0],
                provider_argv0=provider_argv[0],
                cwd=cwd,
                expected_sink=sink,
                expected_executable_identity=executable_identity,
                flow_context=flow_context,
                data_flow_payload=data_flow_payload,
            )
            dispatch_kwargs = dict(provider_kwargs)
            if executable_snapshot is not None:
                dispatch_kwargs["executable_snapshot"] = executable_snapshot
            try:
                try:
                    proc = operation.call(
                        ProviderPhase("run", state_mutation=True, information_flow=True),
                        self.provider.run,
                        provider_argv,
                        **dispatch_kwargs,
                    )
                finally:
                    if executable_snapshot is not None:
                        executable_snapshot.close()
                proc = replace(self._bounded_result(proc), argv=list(checked))
            except Exception as exc:
                self._raise_subprocess_failure(
                    exc,
                    pid=pid,
                    resource=resource,
                    argv=checked,
                    cwd=cwd,
                    timeout=selected_timeout,
                )
                raise

            result_observation = {
                "returncode": proc.returncode,
                "stdout_truncated": proc.stdout_truncated,
                "stderr_truncated": proc.stderr_truncated,
            }
            evidence = ProtectedOperationEvidence(
                event_type=EventType.EXTERNAL_WRITE,
                event_source=pid,
                event_target=resource,
                event_payload={
                    "adapter": "shell",
                    "operation": "run",
                    "argv": checked,
                    "returncode": proc.returncode,
                    "policy_level": decision.policy_level,
                    "high_risk": decision.high_risk,
                    "risk": decision.risk.value,
                    "rule_id": decision.rule_id,
                    "cwd": os.fspath(cwd) if cwd is not None else None,
                },
                audit_action="primitive.shell.run",
                audit_actor=pid,
                audit_target=resource,
                audit_decision={
                    "argv": checked,
                    "returncode": proc.returncode,
                    "policy_level": decision.policy_level,
                    "policy_reason": decision.reason,
                    "matched_rule": list(decision.matched_rule) if decision.matched_rule else None,
                    "high_risk": decision.high_risk,
                    "risk": decision.risk.value,
                    "rule_id": decision.rule_id,
                    "sandbox_profile": self.profile_json(decision.sandbox_profile),
                    "cwd": os.fspath(cwd) if cwd is not None else None,
                    "metrics": self._metrics_json(proc.metrics),
                    "stdout_truncated": proc.stdout_truncated,
                    "stderr_truncated": proc.stderr_truncated,
                },
                correlation_id=correlation_id,
                parent_record_id=intent_record.record_id,
                effect_metadata=result_observation,
            )
            resource_settlement = None
            if proc.metrics is not None:
                resource_settlement = ResourceSettlement(
                    usage=ResourceUsage(
                        subprocess_wall_seconds=max(0.0, proc.metrics.wall_seconds),
                        subprocess_cpu_seconds=max(0.0, proc.metrics.cpu_seconds),
                        subprocess_peak_memory_bytes=max(0, proc.metrics.peak_memory_bytes),
                    ),
                    source="primitive.shell.run",
                    context={
                        "resource": resource,
                        "argv": list(checked),
                        "cwd": os.fspath(cwd) if cwd is not None else None,
                        "metrics": self._metrics_json(proc.metrics),
                    },
                )
            completed = operation.complete(
                proc,
                evidence,
                classification_context=effect_context,
                classification_result=result_observation,
                resource=resource_settlement,
            )
            return completed

    def _raise_subprocess_failure(
        self,
        error: Exception,
        *,
        pid: str,
        resource: str,
        argv: list[str],
        cwd: str | os.PathLike[str] | None,
        timeout: float,
    ) -> None:
        """Translate nested provider timeout/limit failures at one boundary."""

        timeout_error = self._exception_from_cause_chain(
            error,
            (SubprocessTimeoutExpired, subprocess.TimeoutExpired),
        )
        if timeout_error is not None:
            try:
                self._charge_subprocess_metrics(
                    pid,
                    getattr(timeout_error, "metrics", None),
                    resource=resource,
                    argv=argv,
                    cwd=cwd,
                    allow_overage=True,
                )
            except ResourceLimitExceeded:
                # The usage and terminal resource transition are already
                # durable. Preserve the provider timeout as the boundary's
                # explicit classification instead of masking it with the
                # accounting overage raised after charge.
                pass
            raise TimeoutError(
                f"shell command timed out after {timeout}s: {argv}"
            ) from error

        limit_error = self._exception_from_cause_chain(
            error,
            (SubprocessLimitExceeded,),
        )
        if limit_error is None:
            return
        assert isinstance(limit_error, SubprocessLimitExceeded)
        accounting_killed = False
        try:
            self._charge_subprocess_metrics(
                pid,
                limit_error.metrics,
                resource=resource,
                argv=argv,
                cwd=cwd,
                allow_overage=True,
            )
        except ResourceLimitExceeded:
            accounting_killed = True
        reason = str(limit_error)
        if self.resources is not None and not accounting_killed:
            self.resources.kill_if_exceeded(
                pid,
                reason=reason,
                limit={
                    "kind": limit_error.metrics.limit_kind,
                    "metrics": self._metrics_json(limit_error.metrics),
                },
            )
        raise ResourceLimitExceeded(reason) from error

    def harden_read_only_git_argv(self, argv: list[str]) -> list[str]:
        """Apply the shared exact-read Git hardening used by Shell and PTY."""

        return harden_read_only_git_argv(argv)

    def _protected(self):
        return self.protected_operations

    def _data_flow(self) -> Any:
        manager = getattr(self, "data_flow", None) or getattr(
            self._protected(),
            "data_flow",
            None,
        )
        if manager is None:
            raise ValidationError("shell data-flow manager is not attached")
        return manager

    def resolve_provider_argv(
        self,
        argv: list[str],
        *,
        cwd: str | os.PathLike[str] | None,
        provider: Any | None = None,
    ) -> list[str]:
        selected_provider = self.provider if provider is None else provider
        resolver = getattr(selected_provider, "resolve_argv", None)
        if not callable(resolver):
            return list(argv)
        resolved = resolver(
            list(argv),
            cwd=os.fspath(cwd) if cwd is not None else None,
        )
        if isinstance(resolved, (str, bytes)):
            raise ValidationError("provider executable resolver must return argv")
        try:
            selected = list(resolved)
        except TypeError as exc:
            raise ValidationError("provider executable resolver must return argv") from exc
        if (
            len(selected) != len(argv)
            or not selected
            or not isinstance(selected[0], str)
            or not selected[0].strip()
            or selected[1:] != argv[1:]
        ):
            raise ValidationError(
                "provider executable resolver may canonicalize only argv[0]"
            )
        return selected

    def _resolved_executable_identity(
        self,
        argv0: str,
        *,
        cwd: str | os.PathLike[str] | None = None,
    ) -> str:
        """Resolve the Sink identity from Host-owned argv[0] state only."""

        root = self._provider_workspace_root() or self.cwd
        selected_cwd = self._resolve_workspace_cwd(
            root,
            os.fspath(cwd) if cwd is not None else ".",
        )
        if self._argv0_has_path(argv0):
            raw = Path(argv0).expanduser()
            selected = raw if raw.is_absolute() else selected_cwd / raw
        else:
            selected = shutil.which(argv0, path=self._host_executable_path(root))
            if selected is None:
                selected = resolve_runtime_python_alias(
                    argv0,
                    workspace_root=root,
                )
            if selected is None:
                selected = selected_cwd / argv0
        return Path(selected).resolve(strict=False).as_posix()

    def executable_data_sink(
        self,
        namespace: str,
        argv0: str,
        *,
        cwd: str | os.PathLike[str] | None = None,
    ) -> DataSink:
        executable_identity = self._resolved_executable_identity(argv0, cwd=cwd)
        try:
            identity_sha256 = executable_content_sha256(executable_identity)
        except FileNotFoundError:
            # Abstract/fake providers may implement a non-local executable. Such
            # a Sink remains usable for normal data, but cannot match a
            # content-bound clearance above normal.
            identity_sha256 = None
        return DataSink(
            identity=f"{namespace}:{executable_identity}",
            identity_sha256=identity_sha256,
        )

    def _host_executable_path(self, root: Path) -> str:
        entries: list[str] = []
        for item in os.environ.get("PATH", "").split(os.pathsep):
            if not item:
                continue
            raw = Path(item).expanduser()
            if not raw.is_absolute():
                continue
            resolved = raw.resolve(strict=False)
            if root in resolved.parents or resolved == root:
                continue
            entries.append(str(resolved))
        return os.pathsep.join(entries)

    def require_provider_executable_identity(
        self,
        argv0: str,
        *,
        expected: str,
        cwd: str | os.PathLike[str] | None,
    ) -> None:
        actual = self._resolved_executable_identity(argv0, cwd=cwd)
        if actual != expected:
            raise ValidationError(
                "provider executable resolver changed the authorized Sink identity: "
                f"expected {expected}, found {actual}"
            )

    def snapshot_executable_for_dispatch(
        self,
        *,
        pid: str,
        provider: Any,
        requested_argv0: str,
        provider_argv0: str,
        cwd: str | os.PathLike[str] | None,
        expected_sink: DataSink,
        expected_executable_identity: str,
        flow_context: Any,
        data_flow_payload: Any,
    ) -> ExecutableSnapshot | None:
        checker = getattr(provider, "executable_snapshot_required", None)
        if not callable(checker):
            return None
        required = bool(
            checker(
                provider_argv0,
                requested_argv0=requested_argv0,
                cwd=os.fspath(cwd) if cwd is not None else None,
            )
        )
        if not required:
            return None
        if not bool(getattr(provider, "supports_executable_snapshots", False)):
            self._data_flow().reject_sink_identity_change(
                pid=pid,
                sink=expected_sink,
                context=flow_context,
                payload=data_flow_payload,
                reason="provider cannot pin a mutable executable before dispatch",
            )
            raise AssertionError("data-flow Sink rejection must raise")
        snapshot = snapshot_executable(
            provider_argv0,
            sibling_limit=self.config.tools.executable_snapshot_sibling_limit,
        )
        if (
            snapshot.source_path.as_posix() != expected_executable_identity
            or snapshot.content_sha256 != expected_sink.identity_sha256
        ):
            snapshot.close()
            self._data_flow().reject_sink_identity_change(
                pid=pid,
                sink=expected_sink,
                context=flow_context,
                payload=data_flow_payload,
                reason="Sink identity changed before provider dispatch",
            )
            raise AssertionError("data-flow Sink rejection must raise")
        return snapshot

    def _protected_failure_evidence(
        self,
        pid: str,
        resource: str,
        argv: list[str],
        cwd: str | os.PathLike[str] | None,
        intent_record: Any,
        error: BaseException,
        phase: str,
    ) -> ProtectedOperationEvidence:
        classified_error = self._exception_from_cause_chain(
            error,
            (
                SubprocessTimeoutExpired,
                subprocess.TimeoutExpired,
                SubprocessLimitExceeded,
            ),
        ) or error
        if isinstance(classified_error, (SubprocessTimeoutExpired, subprocess.TimeoutExpired)):
            action = "primitive.shell.timeout"
        elif isinstance(classified_error, SubprocessLimitExceeded):
            action = "primitive.shell.resource_limit_exceeded"
        else:
            action = "primitive.shell.failed"
        metrics = self._metrics_json(getattr(classified_error, "metrics", None))
        decision: dict[str, Any] = {
            "argv": list(argv),
            "cwd": os.fspath(cwd) if cwd is not None else None,
            "effect_outcome": "unknown",
            "error_type": type(classified_error).__name__,
            "phase": phase,
        }
        if classified_error is not error:
            decision["wrapper_error_type"] = type(error).__name__
        if metrics is not None:
            decision["metrics"] = metrics
        return ProtectedOperationEvidence(
            event_type=EventType.EXTERNAL_WRITE,
            event_source=pid,
            event_target=resource,
            event_payload={
                "adapter": "shell",
                "operation": "run",
                "outcome": "unknown",
                "error_type": type(classified_error).__name__,
            },
            audit_action=action,
            audit_actor=pid,
            audit_target=resource,
            audit_decision=decision,
            correlation_id=intent_record.record_id,
            parent_record_id=intent_record.record_id,
        )

    @staticmethod
    def _exception_from_cause_chain(
        error: BaseException,
        expected: tuple[type[BaseException], ...],
    ) -> BaseException | None:
        pending: list[BaseException] = [error]
        seen: set[int] = set()
        while pending and len(seen) < 64:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            if isinstance(current, expected):
                return current
            context = current.__context__
            if context is not None and not current.__suppress_context__:
                pending.append(context)
            cause = current.__cause__
            if cause is not None:
                pending.append(cause)
        return None

    async def arun(
        self,
        pid: str,
        argv: list[str],
        timeout: float = _TOOL_DEFAULTS.shell_timeout_s,
        cwd: str | os.PathLike[str] | None = None,
        *,
        source_oids: list[str] | tuple[str, ...] | None = None,
    ) -> CommandResult:
        return await self._data_flow().run_sync_in_worker(
            self.run,
            pid,
            argv,
            timeout=timeout,
            cwd=cwd,
            source_oids=source_oids,
        )

    def grant_policy(
        self,
        pid: str,
        level: ShellPolicyLevel | str | None = None,
        *,
        issued_by: str = "shell",
    ) -> Capability:
        selected = self._normalize_policy_level(level or self.config.shell.default_policy_level)
        return self.capabilities.grant(
            subject=pid,
            resource=self.policy_resource(),
            rights=[CapabilityRight.EXECUTE],
            issued_by=issued_by,
            constraints={self.config.shell.policy_capability_key: selected},
        )

    def policy_resource(self) -> str:
        return self.config.shell.policy_resource

    def resource_for(self, argv: list[str]) -> str:
        command = self._command_identity(argv[0])
        return f"shell:{command}"

    def _authorize(self, pid: str, argv: list[str], resource: str, *, timeout: float, cwd: str) -> ShellPolicyDecision:
        return self.authorize_operation(
            pid,
            argv,
            resource,
            timeout=timeout,
            cwd=cwd,
            adapter="shell",
            primitive="runtime.shell.run",
            operation="shell.run",
            authority_operation="shell.run",
            include_timeout_in_authority=True,
        )

    def authorize_operation(
        self,
        pid: str,
        argv: list[str],
        resource: str,
        *,
        timeout: float,
        cwd: str,
        adapter: str,
        primitive: str,
        operation: str,
        authority_operation: str,
        include_timeout_in_authority: bool,
        continuous_session: bool = False,
        extra_context: dict[str, Any] | None = None,
    ) -> ShellPolicyDecision:
        rule_match, profile, operation_context = self._classify_shell_operation(
            pid,
            argv,
            resource,
            timeout=timeout,
            cwd=cwd,
            adapter=adapter,
            primitive=primitive,
            operation=operation,
            authority_operation=authority_operation,
            include_timeout_in_authority=include_timeout_in_authority,
            continuous_session=continuous_session,
            extra_context=extra_context,
        )
        rule = rule_match.rule
        policy_caps = self._matching_shell_policy_caps(pid, resource, operation_context)
        preflight_denial = self._shell_rule_preflight_denial(
            policy_caps,
            rule_match,
            profile,
        )
        if preflight_denial is not None:
            return preflight_denial

        explicit_decision = self._explicit_command_decision(pid, resource, operation_context)
        explicit_policy = explicit_decision.policy
        if explicit_policy == CapabilityManager.ALWAYS_DENY:
            return ShellPolicyDecision(
                False,
                False,
                "explicit command capability denied command",
                explicit_policy,
                matched_rule=rule_match.matched_argv,
                high_risk=self._is_high_risk(rule.risk),
                risk=rule.risk,
                rule_id=rule.rule_id,
                rule_effect=rule.effect,
                sandbox_profile=profile,
            )
        if explicit_policy == CapabilityManager.ASK_EACH_TIME:
            return ShellPolicyDecision(
                False,
                True,
                "explicit command capability requires approval",
                explicit_policy,
                matched_rule=rule_match.matched_argv,
                high_risk=self._is_high_risk(rule.risk),
                risk=rule.risk,
                rule_id=rule.rule_id,
                rule_effect=rule.effect,
                sandbox_profile=profile,
                authority_decision=explicit_decision,
            )
        if explicit_policy in {CapabilityManager.ALWAYS_ALLOW, CapabilityManager.ALLOW_ONCE}:
            return ShellPolicyDecision(
                allowed=True,
                ask_human=False,
                reason="explicit command capability allowed command",
                policy_level=explicit_policy,
                consume_once=explicit_decision.consume_capability_id is not None,
                consume_capability_id=explicit_decision.consume_capability_id,
                matched_rule=rule_match.matched_argv,
                high_risk=self._is_high_risk(rule.risk),
                risk=rule.risk,
                rule_id=rule.rule_id,
                rule_effect=rule.effect,
                sandbox_profile=profile,
                authority_decision=explicit_decision,
            )

        if not policy_caps:
            raise CapabilityDenied(f"{pid} lacks shell execute policy for {resource}")

        policy_decision = self.capabilities.authorize_matching_capabilities(
            pid,
            resource,
            CapabilityRight.EXECUTE,
            policy_caps,
            operation_context,
        )
        level = self._selected_policy_level(policy_caps, policy_decision.selected_capability_id)
        if policy_decision.effect == CapabilityEffect.ASK:
            return ShellPolicyDecision(
                allowed=False,
                ask_human=True,
                reason="shell policy capability requires approval",
                policy_level=level,
                matched_rule=rule_match.matched_argv,
                high_risk=self._is_high_risk(rule.risk),
                risk=rule.risk,
                rule_id=rule.rule_id,
                rule_effect=rule.effect,
                sandbox_profile=profile,
                authority_decision=policy_decision,
            )
        if not policy_decision.allowed:
            return ShellPolicyDecision(
                allowed=False,
                ask_human=False,
                reason=policy_decision.reason,
                policy_level=level,
                matched_rule=rule_match.matched_argv,
                high_risk=self._is_high_risk(rule.risk),
                risk=rule.risk,
                rule_id=rule.rule_id,
                rule_effect=rule.effect,
                sandbox_profile=profile,
            )
        consume_capability_id = policy_decision.consume_capability_id
        if level in {self.config.shell.allowlist_auto_else_ask_level, self.config.shell.blocklist_ask_else_auto_level}:
            if rule.effect == CapabilityEffect.ALLOW:
                return ShellPolicyDecision(
                    allowed=True,
                    ask_human=False,
                    reason=rule.description or "shell rule allowed command",
                    policy_level=level,
                    consume_once=consume_capability_id is not None,
                    consume_capability_id=consume_capability_id,
                    matched_rule=rule_match.matched_argv,
                    high_risk=self._is_high_risk(rule.risk),
                    risk=rule.risk,
                    rule_id=rule.rule_id,
                    rule_effect=rule.effect,
                    sandbox_profile=profile,
                    authority_decision=policy_decision,
                )
            return ShellPolicyDecision(
                allowed=False,
                ask_human=True,
                reason=rule.description or "shell rule requires approval",
                policy_level=level,
                matched_rule=rule_match.matched_argv,
                high_risk=self._is_high_risk(rule.risk),
                risk=rule.risk,
                rule_id=rule.rule_id,
                rule_effect=rule.effect,
                sandbox_profile=profile,
                authority_decision=policy_decision,
            )
        if level == self.config.shell.always_allow_level:
            return ShellPolicyDecision(
                allowed=True,
                ask_human=False,
                reason="shell policy is always_allow",
                policy_level=level,
                consume_once=consume_capability_id is not None,
                consume_capability_id=consume_capability_id,
                matched_rule=rule_match.matched_argv,
                high_risk=self._is_high_risk(rule.risk),
                risk=rule.risk,
                rule_id=rule.rule_id,
                rule_effect=rule.effect,
                sandbox_profile=profile,
                authority_decision=policy_decision,
            )
        return ShellPolicyDecision(
            False,
            False,
            f"unsupported shell policy level: {level}",
            level,
            matched_rule=rule_match.matched_argv,
            high_risk=self._is_high_risk(rule.risk),
            risk=rule.risk,
            rule_id=rule.rule_id,
            rule_effect=rule.effect,
            sandbox_profile=profile,
        )

    def _shell_rule_preflight_denial(
        self,
        policy_caps: list[Capability],
        rule_match: RuleMatch,
        profile: SandboxProfile,
    ) -> ShellPolicyDecision | None:
        rule = rule_match.rule
        if any(
            cap.effect == CapabilityEffect.DENY
            or
            cap.constraints.get(self.config.shell.policy_capability_key) == self.config.shell.always_deny_level
            for cap in policy_caps
        ):
            return ShellPolicyDecision(
                allowed=False,
                ask_human=False,
                reason="shell policy is always_deny",
                policy_level=self.config.shell.always_deny_level,
                matched_rule=rule_match.matched_argv,
                risk=rule.risk,
                rule_id=rule.rule_id,
                rule_effect=rule.effect,
                sandbox_profile=profile,
            )
        if rule.effect == CapabilityEffect.DENY:
            return ShellPolicyDecision(
                allowed=False,
                ask_human=False,
                reason=rule.description or "shell rule denied command",
                policy_level=None,
                matched_rule=rule_match.matched_argv,
                high_risk=True,
                risk=rule.risk,
                rule_id=rule.rule_id,
                rule_effect=rule.effect,
                sandbox_profile=profile,
            )
        return None

    def _classify_shell_operation(
        self,
        pid: str,
        argv: list[str],
        resource: str,
        *,
        timeout: float,
        cwd: str,
        adapter: str,
        primitive: str,
        operation: str,
        authority_operation: str,
        include_timeout_in_authority: bool,
        continuous_session: bool,
        extra_context: dict[str, Any] | None,
    ) -> tuple[RuleMatch, SandboxProfile, dict[str, Any]]:
        builtin_rule = self.rule_engine.classify_builtin(argv).rule
        provisional_profile = self._profile_for_shell_rule(
            rule=builtin_rule,
            resource=resource,
            argv=argv,
            timeout=timeout,
            cwd=cwd,
            operation=operation,
            include_timeout_in_authority=include_timeout_in_authority,
            continuous_session=continuous_session,
        )
        classification_context = self.operation_context(
            pid,
            argv,
            resource,
            timeout=timeout,
            cwd=cwd,
            profile=provisional_profile,
            adapter=adapter,
            primitive=primitive,
            operation=operation,
            authority_operation=authority_operation,
            include_timeout=include_timeout_in_authority,
            continuous_session=continuous_session,
            extra=extra_context,
        )
        # Classification conditions always see the requested launch timeout
        # and the explicit session mode, even when those fields are omitted
        # from the capability authority projection (for example PTY startup).
        classification_context["timeout_s"] = timeout
        classification_context["continuous_session"] = continuous_session
        rule_match = self.rule_engine.classify(argv, context=classification_context)
        profile = self._profile_for_shell_rule(
            rule=rule_match.rule,
            resource=resource,
            argv=argv,
            timeout=timeout,
            cwd=cwd,
            operation=operation,
            include_timeout_in_authority=include_timeout_in_authority,
            continuous_session=continuous_session,
        )
        operation_context = self.operation_context(
            pid,
            argv,
            resource,
            timeout=timeout,
            cwd=cwd,
            profile=profile,
            adapter=adapter,
            primitive=primitive,
            operation=operation,
            authority_operation=authority_operation,
            include_timeout=include_timeout_in_authority,
            continuous_session=continuous_session,
            extra=extra_context,
        )
        return rule_match, profile, operation_context

    def _profile_for_shell_rule(
        self,
        *,
        rule: AuthorityRule,
        resource: str,
        argv: list[str],
        timeout: float,
        cwd: str,
        operation: str,
        include_timeout_in_authority: bool,
        continuous_session: bool,
    ) -> SandboxProfile:
        profile = self.capabilities.profiles.shell(
            resource=resource,
            effect=rule.effect,
            risk=rule.risk,
            rule_id=rule.rule_id,
            argv=argv,
            timeout_s=timeout,
            cwd=cwd,
        )
        if operation != "shell.run" or continuous_session or not include_timeout_in_authority:
            restrictions = dict(profile.restrictions)
            if not include_timeout_in_authority:
                restrictions.pop("timeout_s", None)
                restrictions["startup_timeout_s"] = timeout
            if continuous_session:
                restrictions["continuous_session"] = True
            profile = SandboxProfile(
                operation=operation,
                resource=resource,
                effect=rule.effect,
                risk=rule.risk,
                rule_id=rule.rule_id,
                restrictions=restrictions,
            )
        return profile

    def _request_human_approval(
        self,
        pid: str,
        argv: list[str],
        resource: str,
        decision: ShellPolicyDecision,
        *,
        timeout: float,
        cwd: str | os.PathLike[str] | None,
        source_oids: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        if self.human is None:
            raise CapabilityDenied(f"{pid} requires human approval for shell execute on {resource}")
        selected_cwd = os.fspath(cwd) if cwd is not None else "."
        approval_context = self.operation_context(
            pid,
            argv,
            resource,
            timeout=timeout,
            cwd=selected_cwd,
            profile=decision.sandbox_profile,
        )
        approval_context.update(
            {
                "workspace_root": str(getattr(self.provider, "cwd", "")),
                "working_directory": selected_cwd,
                "grant_scope": "one_time",
                "policy_level": decision.policy_level,
                "policy_reason": decision.reason,
                "matched_rule": list(decision.matched_rule) if decision.matched_rule else None,
                "high_risk": decision.high_risk,
            }
        )
        request_id = self.human.query(
            pid=pid,
            human=self.config.runtime.default_human,
            request={
                "type": "external_operation_approval",
                "question": f"Allow this process to run shell command {argv[0]!r}?",
                "requested_once_capability": {
                    "subject": pid,
                    "resource": resource,
                    "rights": [CapabilityRight.EXECUTE.value],
                    "constraints": self.approval_constraints(argv, decision, timeout=timeout, cwd=os.fspath(cwd) if cwd is not None else "."),
                },
                "context": approval_context,
            },
            blocking=True,
            source_oids=source_oids,
        )
        raise HumanApprovalRequired(
            request_id=request_id,
            message=f"{pid} is waiting for per-use human approval to run {resource}",
        )

    def _record_run_intent(
        self,
        pid: str,
        resource: str,
        argv: list[str],
        decision: ShellPolicyDecision,
        *,
        timeout: float,
        cwd: str | os.PathLike[str] | None,
    ) -> Any:
        return self.audit.record(
            actor=pid,
            action="primitive.shell.intent",
            target=resource,
            decision={
                "argv": argv,
                "timeout_s": timeout,
                "policy_level": decision.policy_level,
                "policy_reason": decision.reason,
                "matched_rule": list(decision.matched_rule) if decision.matched_rule else None,
                "high_risk": decision.high_risk,
                "risk": decision.risk.value,
                "rule_id": decision.rule_id,
                "sandbox_profile": self.profile_json(decision.sandbox_profile),
                "cwd": os.fspath(cwd) if cwd is not None else None,
            },
        )

    def _configured_rules(self) -> list[AuthorityRule]:
        rules: list[AuthorityRule] = list(self.config.shell.rules)
        for rule in self.config.shell.whitelist:
            rules.append(
                AuthorityRule(
                    rule_id=f"shell.config.allow.{'.'.join(rule.argv)}",
                    operation="shell.run",
                    effect=CapabilityEffect.ALLOW,
                    risk=AuthorityRisk.HARMLESS,
                    conditions={"argv": list(rule.argv), "match": rule.match},
                    description=rule.description or "configured harmless shell allow rule",
                )
            )
        for rule in self.config.shell.blacklist:
            rules.append(
                AuthorityRule(
                    rule_id=f"shell.config.ask.{'.'.join(rule.argv)}",
                    operation="shell.run",
                    effect=CapabilityEffect.ASK,
                    risk=AuthorityRisk.HIGH,
                    conditions={"argv": list(rule.argv), "match": rule.match},
                    description=rule.description or "configured high-risk shell ask rule",
                )
            )
        return rules

    def _is_high_risk(self, risk: AuthorityRisk) -> bool:
        return risk in {AuthorityRisk.HIGH, AuthorityRisk.DESTRUCTIVE}

    def profile_json(self, profile: SandboxProfile | None) -> dict[str, Any] | None:
        if profile is None:
            return None
        return {
            "operation": profile.operation,
            "resource": profile.resource,
            "effect": profile.effect.value,
            "risk": profile.risk.value,
            "rule_id": profile.rule_id,
            "restrictions": profile.restrictions,
        }

    def operation_context(
        self,
        pid: str,
        argv: list[str],
        resource: str,
        *,
        timeout: float,
        cwd: str,
        profile: SandboxProfile,
        adapter: str = "shell",
        primitive: str = "runtime.shell.run",
        operation: str = "shell.run",
        authority_operation: str = "shell.run",
        include_timeout: bool = True,
        continuous_session: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        argv_json = "\0".join(argv)
        profile_json = self.profile_json(profile)
        context = {
            "adapter": adapter,
            "primitive": primitive,
            "operation": operation,
            "authority_operation": authority_operation,
            "pid": pid,
            "argv": list(argv),
            "argv_sha256": hashlib.sha256(argv_json.encode("utf-8")).hexdigest(),
            "command": argv[0],
            "resource": resource,
            "right": CapabilityRight.EXECUTE.value,
            "cwd": cwd,
            "risk": profile.risk.value,
            "rule_id": profile.rule_id,
            "rule_effect": profile.effect.value,
            "sandbox_profile": profile_json,
            "network": bool(profile_json and profile_json["restrictions"].get("network")),
            "filesystem_intent": profile_json["restrictions"].get("filesystem_intent") if profile_json else None,
        }
        if include_timeout:
            context["timeout_s"] = timeout
        if continuous_session:
            context["continuous_session"] = True
        if extra:
            context.update(extra)
        return context

    def approval_constraints(
        self,
        argv: list[str],
        decision: ShellPolicyDecision,
        *,
        timeout: float,
        cwd: str,
        operation: str = "shell.run",
        include_timeout: bool = True,
        extra_conditions: dict[str, Any] | None = None,
        description: str = "one-shot human approval for exact shell argv",
    ) -> dict[str, Any]:
        argv_json = "\0".join(argv)
        conditions: dict[str, Any] = {
            "argv": list(argv),
            "match": "exact",
            "argv_sha256": hashlib.sha256(argv_json.encode("utf-8")).hexdigest(),
            "cwd": cwd,
        }
        if include_timeout:
            conditions["timeout_s"] = timeout
        if extra_conditions:
            conditions.update(extra_conditions)
        return {
            AUTHORITY_RULES_KEY: [
                {
                    "rule_id": f"shell.approval.{decision.rule_id or 'exact'}",
                    "operation": operation,
                    "effect": CapabilityEffect.ALLOW.value,
                    "risk": decision.risk.value,
                    "conditions": conditions,
                    "description": description,
                }
            ]
        }

    def _matching_shell_policy_caps(self, pid: str, resource: str, context: dict[str, Any]) -> list[Capability]:
        caps = [
            cap
            for cap in self.capabilities.matching_capabilities(
                pid,
                resource,
                CapabilityRight.EXECUTE,
                include_ask=True,
            )
            if self.config.shell.policy_capability_key in cap.constraints
        ]
        caps = [
            cap
            for cap in caps
            if cap.effect == CapabilityEffect.DENY
            or self.capabilities.constraints_satisfied(cap, context)
        ]
        caps.sort(key=lambda cap: (len(cap.resource), cap.issued_at), reverse=True)
        return caps

    def _explicit_command_decision(self, pid: str, resource: str, context: dict[str, Any]) -> Any:
        """Classify direct `shell:<command>` authority separate from policy caps.

        A shell policy capability decides how whitelist/blacklist rules are
        applied. A direct command capability is narrower authority granted by
        human approval or bootstrap. Both are canonical Capability records and both
        use the central resource matcher; they are split only so a broad
        `shell:*` policy does not bypass shell-specific token checks.
        """

        caps = [
            cap
            for cap in self.capabilities.matching_capabilities(
                pid,
                resource,
                CapabilityRight.EXECUTE,
                include_ask=True,
            )
            if self.config.shell.policy_capability_key not in cap.constraints
        ]
        # A broad `shell:*` allow is a policy grant only when it carries
        # shell_policy_level. Treating it as direct command authority would turn
        # a registry-level wildcard into an unreviewed "always allow" shell.
        caps = [
            cap
            for cap in caps
            if cap.resource != self.config.shell.policy_resource or cap.effect != CapabilityEffect.ALLOW
        ]
        caps = [
            cap
            for cap in caps
            if self._direct_shell_capability_applies_to_rule(cap, context)
        ]
        if not caps:
            return self.capabilities.authorize_matching_capabilities(pid, resource, CapabilityRight.EXECUTE, [], context)
        caps.sort(key=lambda cap: (len(cap.resource), cap.issued_at), reverse=True)
        return self.capabilities.authorize_matching_capabilities(pid, resource, CapabilityRight.EXECUTE, caps, context)

    def _direct_shell_capability_applies_to_rule(self, cap: Capability, context: dict[str, Any]) -> bool:
        if context.get("authority_operation", "shell.run") != "shell.run":
            if cap.effect != CapabilityEffect.ALLOW:
                return True
            return AUTHORITY_RULES_KEY in cap.constraints
        if cap.effect != CapabilityEffect.ALLOW:
            return True
        if AUTHORITY_RULES_KEY in cap.constraints:
            return True
        # Bare `shell:<command>` grants are command-family hints, not permission
        # to run every subcommand. Without AuthorityRule constraints they only
        # cover argv that the deterministic classifier already considers
        # harmless/low-risk allow; medium/high commands need an explicit rule or
        # a shell policy approved by the human.
        return context.get("rule_effect") == CapabilityEffect.ALLOW.value

    def _selected_policy_level(self, caps: list[Capability], selected_capability_id: str | None = None) -> str:
        selected = next(
            (cap for cap in caps if cap.cap_id == selected_capability_id),
            caps[0],
        )
        return self._normalize_policy_level(selected.constraints[self.config.shell.policy_capability_key])

    def _normalize_policy_level(self, value: Any) -> str:
        normalized = str(value).strip().lower()
        allowed = {
            self.config.shell.always_deny_level,
            self.config.shell.allowlist_auto_else_ask_level,
            self.config.shell.blocklist_ask_else_auto_level,
            self.config.shell.always_allow_level,
        }
        if normalized not in allowed:
            raise ValidationError(f"unknown shell policy level: {value!r}")
        return normalized

    def validate_argv(self, argv: list[str]) -> list[str]:
        if not isinstance(argv, list) or not argv:
            raise ValidationError("shell argv must be a non-empty list")
        checked: list[str] = []
        for index, value in enumerate(argv):
            if not isinstance(value, str):
                raise ValidationError(f"shell argv[{index}] must be a string")
            if "\x00" in value:
                raise ValidationError(f"shell argv[{index}] cannot contain NUL bytes")
            if index == 0 and not value.strip():
                raise ValidationError("shell argv[0] must be non-empty")
            checked.append(value)
        # This direct-argv check deliberately precedes policy evaluation.
        # General interpreters remain governed by Shell authority because the
        # local provider is not an operating-system subprocess sandbox.
        return validate_and_normalize_raw_git(checked)

    def _validate_timeout(self, timeout: float) -> float:
        try:
            selected = float(timeout)
        except (TypeError, ValueError) as exc:
            raise ValidationError("shell timeout must be a number") from exc
        if not math.isfinite(selected):
            raise ValidationError("shell timeout must be finite")
        if selected <= 0:
            raise ValidationError("shell timeout must be > 0")
        if selected > self.config.shell.timeout_hard_limit_s:
            raise ValidationError(f"shell timeout exceeds hard limit {self.config.shell.timeout_hard_limit_s}s")
        return selected

    def _bounded_result(self, proc: CommandResult) -> CommandResult:
        stdout, stdout_truncated = self._truncate_output(proc.stdout, self.config.shell.max_stdout_chars)
        stderr, stderr_truncated = self._truncate_output(proc.stderr, self.config.shell.max_stderr_chars)
        return CommandResult(
            argv=proc.argv,
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=proc.stdout_truncated or stdout_truncated,
            stderr_truncated=proc.stderr_truncated or stderr_truncated,
            metrics=proc.metrics,
        )

    def _provider_run_kwargs(
        self,
        *,
        timeout: float,
        cwd: str | os.PathLike[str] | None,
        limits: SubprocessLimits | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"timeout": timeout}
        if cwd is not None:
            kwargs["cwd"] = os.fspath(cwd)
        if limits is not None:
            if not bool(getattr(self.provider, "supports_subprocess_limits", False)):
                raise ValidationError("shell provider must explicitly support SubprocessLimits before budgeted execution")
            kwargs["limits"] = limits
        kwargs["stdout_limit_chars"] = self.config.shell.stdout_hard_limit_chars
        kwargs["stderr_limit_chars"] = self.config.shell.stderr_hard_limit_chars
        self._require_provider_run_parameters_support(kwargs)
        return kwargs

    def _require_provider_run_parameters_support(self, kwargs: dict[str, Any]) -> None:
        try:
            signature = inspect.signature(self.provider.run)
        except (TypeError, ValueError) as exc:
            raise ValidationError("shell provider must expose a signature that accepts shell execution controls") from exc
        supports_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if supports_kwargs:
            return
        missing = sorted(key for key in kwargs if key not in signature.parameters and key not in {"timeout", "cwd"})
        if missing:
            if "limits" in missing:
                raise ValidationError("shell provider must accept SubprocessLimits when resource limits are configured")
            raise ValidationError(f"shell provider must accept execution control parameters: {missing}")

    def subprocess_limits(self, pid: str) -> SubprocessLimits | None:
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
            raise ResourceLimitExceeded(f"process {pid} exhausted subprocess wall-time budget")
        if cpu is not None and cpu <= 0:
            raise ResourceLimitExceeded(f"process {pid} exhausted subprocess CPU budget")
        if memory is not None and memory <= 0:
            raise ResourceLimitExceeded(f"process {pid} exhausted subprocess memory budget")
        if wall is None and cpu is None and memory is None:
            return None
        return SubprocessLimits(wall_seconds=wall, cpu_seconds=cpu, memory_bytes=memory)

    def _charge_subprocess_metrics(
        self,
        pid: str,
        metrics: CommandMetrics | None,
        *,
        resource: str,
        argv: list[str],
        cwd: str | os.PathLike[str] | None,
        allow_overage: bool,
    ) -> None:
        if self.resources is None or metrics is None:
            return
        self.resources.charge(
            pid,
            ResourceUsage(
                subprocess_wall_seconds=max(0.0, metrics.wall_seconds),
                subprocess_cpu_seconds=max(0.0, metrics.cpu_seconds),
                subprocess_peak_memory_bytes=max(0, metrics.peak_memory_bytes),
            ),
            source="primitive.shell.run",
            context={
                "resource": resource,
                "argv": list(argv),
                "cwd": os.fspath(cwd) if cwd is not None else None,
                "metrics": self._metrics_json(metrics),
            },
            allow_overage=allow_overage,
            kill_on_exceed=True,
        )

    def _metrics_json(self, metrics: CommandMetrics | None) -> dict[str, Any] | None:
        if metrics is None:
            return None
        return {
            "wall_seconds": metrics.wall_seconds,
            "cpu_seconds": metrics.cpu_seconds,
            "peak_memory_bytes": metrics.peak_memory_bytes,
            "killed": metrics.killed,
            "limit_kind": metrics.limit_kind,
        }

    def _truncate_output(self, value: str, limit: int) -> tuple[str, bool]:
        if len(value) <= limit:
            return value, False
        return value[:limit], True

    def _command_identity(self, argv0: str) -> str:
        if self._argv0_has_path(argv0):
            return argv0.strip().replace("\\", "/").casefold()
        return self._normalize_executable(argv0)

    def _normalize_executable(self, value: str) -> str:
        raw = value.strip().replace("\\", "/")
        name = PurePath(raw).name or raw
        lowered = name.casefold()
        for suffix in _WINDOWS_EXECUTABLE_SUFFIXES:
            if lowered.endswith(suffix):
                return lowered[: -len(suffix)]
        return lowered

    def _argv0_has_path(self, value: str) -> bool:
        return "/" in value or "\\" in value or PurePath(value).is_absolute()

    def enforce_workspace_argv_scope(self, argv: list[str], *, cwd: str) -> None:
        root = self._provider_workspace_root()
        if root is None:
            return
        selected_cwd = self._resolve_workspace_cwd(root, cwd)
        for token in self._argv_path_tokens(argv):
            target = self._resolve_argument_path(token, root=root, cwd=selected_cwd)
            if root not in target.parents and target != root:
                raise CapabilityDenied(f"shell argument path escapes workspace root: {token}")

    def _provider_workspace_root(self) -> Path | None:
        provider_cwd = getattr(self.provider, "cwd", None)
        if provider_cwd is None:
            return None
        try:
            return Path(provider_cwd).resolve()
        except (OSError, RuntimeError):
            return None

    def _resolve_workspace_cwd(self, root: Path, cwd: str) -> Path:
        if cwd in {"", "."}:
            return root
        raw = Path(cwd)
        target = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
        if root not in target.parents and target != root:
            raise CapabilityDenied(f"shell working directory escapes workspace root: {cwd}")
        return target

    def _argv_path_tokens(self, argv: list[str]) -> list[str]:
        tokens: list[str] = []
        for index, value in enumerate(argv):
            candidates = [value]
            if index > 0 and value.startswith("-") and "=" in value:
                candidates = [value.split("=", 1)[1]]
            elif index > 0 and value.startswith("-") and not value.startswith("--"):
                attached = self._attached_short_option_path(value)
                candidates = [] if attached is None else [attached]
            for candidate in candidates:
                if self._is_path_like_argument(candidate, argv0=index == 0):
                    tokens.append(candidate)
        return tokens

    def _attached_short_option_path(self, value: str) -> str | None:
        if len(value) <= 2:
            return None
        lowered = value.casefold()
        if "://" in lowered and "file://" not in lowered:
            return None
        operand = value[2:]
        normalized_operand = operand.replace("\\", "/")
        if operand in {".", "..", "~"} or normalized_operand.startswith(("./", "../", "~/", "/")):
            return operand
        if self._is_absolute_path(operand):
            return operand
        normalized = value.replace("\\", "/")
        for marker in ("../", "./", "~/", "/"):
            position = normalized.find(marker, 2)
            if position >= 0:
                return value[position:]
        return None

    def _is_path_like_argument(self, value: str, *, argv0: bool) -> bool:
        if not value:
            return False
        if self._is_file_url(value):
            return True
        if self._looks_like_url(value):
            return False
        if argv0:
            return self._argv0_has_path(value)
        normalized = value.replace("\\", "/")
        if normalized.startswith(("~", "./", "../")) or normalized in {".", ".."}:
            return True
        if self._is_absolute_path(value):
            return True
        return "/" in value or "\\" in value

    def _looks_like_url(self, value: str) -> bool:
        parsed = urlsplit(value)
        return bool(parsed.scheme and parsed.netloc)

    def _is_file_url(self, value: str) -> bool:
        return urlsplit(value).scheme.casefold() == "file"

    def _is_absolute_path(self, value: str) -> bool:
        return Path(value).is_absolute() or PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()

    def _resolve_argument_path(self, value: str, *, root: Path, cwd: Path) -> Path:
        if self._is_file_url(value):
            raise CapabilityDenied(f"shell argument path uses file URL syntax: {value}")
        if value.startswith("~"):
            raise CapabilityDenied(f"shell argument path uses host home expansion: {value}")
        if PureWindowsPath(value).is_absolute() and not Path(value).is_absolute():
            raise CapabilityDenied(f"shell argument path uses host absolute path syntax: {value}")
        raw = Path(value.replace("\\", os.sep))
        if self._is_absolute_path(value):
            return raw.resolve()
        return (cwd / raw).resolve()
