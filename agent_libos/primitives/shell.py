from __future__ import annotations

import os
import asyncio
import hashlib
import inspect
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

from agent_libos.capability.manager import CapabilityManager
from agent_libos.capability.rules import AUTHORITY_RULES_KEY, ShellRuleEngine
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig, ShellCommandRule, ShellPolicyLevel
from agent_libos.models import AuthorityRisk, AuthorityRule, Capability, CapabilityEffect, CapabilityRight, EventType, ResourceUsage, SandboxProfile
from agent_libos.models.exceptions import CapabilityDenied, HumanApprovalRequired, ResourceLimitExceeded, ValidationError
from agent_libos.runtime.audit_manager import AuditManager
from agent_libos.runtime.event_bus import EventBus
from agent_libos.runtime.external_effects import (
    classify_external_effect,
    record_external_effect,
    require_external_effect_classifier,
)
from agent_libos.substrate import (
    CommandMetrics,
    CommandResult,
    LocalShellProvider,
    ShellProvider,
    SubprocessLimitExceeded,
    SubprocessLimits,
    SubprocessTimeoutExpired,
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


class ShellAdapter:
    """Capability-checked shell primitive.

    Commands are accepted only as argv arrays and are executed by the substrate
    with shell=False. Allow/ask decisions use exact token rules; no glob,
    substring, or shell-style parsing is used for whitelist matching.
    """

    def __init__(
        self,
        capabilities: CapabilityManager,
        audit: AuditManager,
        events: EventBus | None = None,
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
        self.human = human
        self.resources = resources
        self.provider = provider or LocalShellProvider(cwd or ".")
        self.rule_engine = ShellRuleEngine(self._configured_rules())

    def run(
        self,
        pid: str,
        argv: list[str],
        timeout: float = _TOOL_DEFAULTS.shell_timeout_s,
        cwd: str | os.PathLike[str] | None = None,
    ) -> CommandResult:
        checked = self._validate_argv(argv)
        selected_timeout = self._validate_timeout(timeout)
        resource = self.resource_for(checked)
        selected_cwd = os.fspath(cwd) if cwd is not None else "."
        self._enforce_workspace_argv_scope(checked, cwd=selected_cwd)
        decision = self._authorize(pid, checked, resource, timeout=selected_timeout, cwd=selected_cwd)
        if decision.ask_human:
            self._request_human_approval(pid, checked, resource, decision, timeout=selected_timeout, cwd=cwd)
        if not decision.allowed:
            raise CapabilityDenied(f"{pid} denied shell execute on {resource}: {decision.reason}")
        effect_context = {
            "argv": list(checked),
            "resource": resource,
            "timeout_s": selected_timeout,
            "cwd": os.fspath(cwd) if cwd is not None else None,
            "policy_level": decision.policy_level,
            "high_risk": decision.high_risk,
            "risk": decision.risk.value,
            "rule_id": decision.rule_id,
            "rule_effect": decision.rule_effect.value,
            "sandbox_profile": self._profile_json(decision.sandbox_profile),
        }
        limits = self._subprocess_limits(pid)
        require_external_effect_classifier(self.provider, "run")
        intent_record = self._record_run_intent(
            pid,
            resource,
            checked,
            decision,
            timeout=selected_timeout,
            cwd=cwd,
        )
        correlation_id = intent_record.record_id
        if decision.consume_once and decision.consume_capability_id is not None:
            self.capabilities.consume_use(
                decision.consume_capability_id,
                used_by="shell",
                reason="one-time shell permission consumed",
            )
        try:
            proc = self._provider_run(checked, timeout=selected_timeout, cwd=cwd, limits=limits)
            proc = self._bounded_result(proc)
        except SubprocessTimeoutExpired as exc:
            charge_error: ResourceLimitExceeded | None = None
            try:
                self._charge_subprocess_metrics(
                    pid,
                    exc.metrics,
                    resource=resource,
                    argv=checked,
                    cwd=cwd,
                    allow_overage=True,
                )
            except ResourceLimitExceeded as resource_exc:
                charge_error = resource_exc
            self.audit.record(
                actor=pid,
                action="primitive.shell.timeout",
                target=resource,
                decision={
                    "argv": checked,
                    "timeout_s": selected_timeout,
                    "cwd": os.fspath(cwd) if cwd is not None else None,
                    "metrics": self._metrics_json(exc.metrics),
                },
                correlation_id=correlation_id,
                parent_record_id=intent_record.record_id,
            )
            if charge_error is not None:
                raise charge_error from exc
            raise TimeoutError(f"shell command timed out after {selected_timeout}s: {checked}") from exc
        except subprocess.TimeoutExpired as exc:
            self.audit.record(
                actor=pid,
                action="primitive.shell.timeout",
                target=resource,
                decision={"argv": checked, "timeout_s": selected_timeout, "cwd": os.fspath(cwd) if cwd is not None else None},
                correlation_id=correlation_id,
                parent_record_id=intent_record.record_id,
            )
            raise TimeoutError(f"shell command timed out after {selected_timeout}s: {checked}") from exc
        except SubprocessLimitExceeded as exc:
            metrics = exc.metrics
            charge_error: ResourceLimitExceeded | None = None
            try:
                self._charge_subprocess_metrics(
                    pid,
                    metrics,
                    resource=resource,
                    argv=checked,
                    cwd=cwd,
                    allow_overage=True,
                )
            except ResourceLimitExceeded as resource_exc:
                charge_error = resource_exc
            reason = str(exc)
            if self.resources is not None:
                self.resources.kill_if_exceeded(
                    pid,
                    reason=reason,
                    limit={"kind": metrics.limit_kind, "metrics": self._metrics_json(metrics)},
                )
            self.audit.record(
                actor=pid,
                action="primitive.shell.resource_limit_exceeded",
                target=resource,
                decision={
                    "argv": checked,
                    "cwd": os.fspath(cwd) if cwd is not None else None,
                    "metrics": self._metrics_json(metrics),
                    "reason": reason,
                },
                correlation_id=correlation_id,
                parent_record_id=intent_record.record_id,
            )
            raise (charge_error or ResourceLimitExceeded(reason)) from exc
        except Exception as exc:
            self.audit.record(
                actor=pid,
                action="primitive.shell.failed",
                target=resource,
                decision={
                    "argv": checked,
                    "cwd": os.fspath(cwd) if cwd is not None else None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                correlation_id=correlation_id,
                parent_record_id=intent_record.record_id,
            )
            raise
        event = self._emit_run_event(pid, resource, checked, proc, decision, cwd=cwd, correlation_id=correlation_id)
        audit_record = self.audit.record(
            actor=pid,
            action="primitive.shell.run",
            target=resource,
            decision={
                "argv": checked,
                "returncode": proc.returncode,
                "policy_level": decision.policy_level,
                "policy_reason": decision.reason,
                "matched_rule": list(decision.matched_rule) if decision.matched_rule else None,
                "high_risk": decision.high_risk,
                "risk": decision.risk.value,
                "rule_id": decision.rule_id,
                "sandbox_profile": self._profile_json(decision.sandbox_profile),
                "cwd": os.fspath(cwd) if cwd is not None else None,
                "metrics": self._metrics_json(proc.metrics),
                "stdout_truncated": proc.stdout_truncated,
                "stderr_truncated": proc.stderr_truncated,
            },
            correlation_id=correlation_id,
            parent_record_id=intent_record.record_id,
        )
        classification = classify_external_effect(
            self.provider,
            "run",
            effect_context,
            {"returncode": proc.returncode, "stdout_truncated": proc.stdout_truncated, "stderr_truncated": proc.stderr_truncated},
        )
        record_external_effect(
            self.audit.store,
            pid=pid,
            provider="shell",
            operation="run",
            target=resource,
            classification=classification,
            audit_record=audit_record,
            event=event,
            metadata={"context": effect_context, "returncode": proc.returncode},
        )
        self._charge_subprocess_metrics(
            pid,
            proc.metrics,
            resource=resource,
            argv=checked,
            cwd=cwd,
            allow_overage=True,
        )
        return proc

    async def arun(
        self,
        pid: str,
        argv: list[str],
        timeout: float = _TOOL_DEFAULTS.shell_timeout_s,
        cwd: str | os.PathLike[str] | None = None,
    ) -> CommandResult:
        return await asyncio.to_thread(self.run, pid, argv, timeout=timeout, cwd=cwd)

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
        return self._authorize_operation(
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

    def _authorize_operation(
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
        rule_match = self.rule_engine.classify(argv)
        rule = rule_match.rule
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
        operation_context = self._operation_context(
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
        policy_caps = self._matching_shell_policy_caps(pid, resource, operation_context)
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
            )

        if not policy_caps:
            raise CapabilityDenied(f"{pid} lacks shell execute policy for {resource}")

        level = self._selected_policy_level(policy_caps)
        if level in {self.config.shell.allowlist_auto_else_ask_level, self.config.shell.blocklist_ask_else_auto_level}:
            if rule.effect == CapabilityEffect.ALLOW:
                return ShellPolicyDecision(
                    allowed=True,
                    ask_human=False,
                    reason=rule.description or "shell rule allowed command",
                    policy_level=level,
                    matched_rule=rule_match.matched_argv,
                    high_risk=self._is_high_risk(rule.risk),
                    risk=rule.risk,
                    rule_id=rule.rule_id,
                    rule_effect=rule.effect,
                    sandbox_profile=profile,
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
            )
        if level == self.config.shell.always_allow_level:
            return ShellPolicyDecision(
                allowed=True,
                ask_human=False,
                reason="shell policy is always_allow",
                policy_level=level,
                matched_rule=rule_match.matched_argv,
                high_risk=self._is_high_risk(rule.risk),
                risk=rule.risk,
                rule_id=rule.rule_id,
                rule_effect=rule.effect,
                sandbox_profile=profile,
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

    def _request_human_approval(
        self,
        pid: str,
        argv: list[str],
        resource: str,
        decision: ShellPolicyDecision,
        *,
        timeout: float,
        cwd: str | os.PathLike[str] | None,
    ) -> None:
        if self.human is None:
            raise CapabilityDenied(f"{pid} requires human approval for shell execute on {resource}")
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
                    "constraints": self._approval_constraints(argv, decision, timeout=timeout, cwd=os.fspath(cwd) if cwd is not None else "."),
                },
                "context": {
                    "adapter": "shell",
                    "primitive": "runtime.shell.run",
                    "operation": "run",
                    "pid": pid,
                    "workspace_root": str(getattr(self.provider, "cwd", "")),
                    "working_directory": os.fspath(cwd) if cwd is not None else ".",
                    "argv": list(argv),
                    "command": argv[0],
                    "resource": resource,
                    "right": CapabilityRight.EXECUTE.value,
                    "grant_scope": "one_time",
                    "timeout_s": timeout,
                    "policy_level": decision.policy_level,
                    "policy_reason": decision.reason,
                    "matched_rule": list(decision.matched_rule) if decision.matched_rule else None,
                    "high_risk": decision.high_risk,
                    "risk": decision.risk.value,
                    "rule_id": decision.rule_id,
                    "rule_effect": decision.rule_effect.value,
                    "sandbox_profile": self._profile_json(decision.sandbox_profile),
                },
            },
            blocking=True,
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
                "sandbox_profile": self._profile_json(decision.sandbox_profile),
                "cwd": os.fspath(cwd) if cwd is not None else None,
            },
        )

    def _emit_run_event(
        self,
        pid: str,
        resource: str,
        argv: list[str],
        proc: CommandResult,
        decision: ShellPolicyDecision,
        *,
        cwd: str | os.PathLike[str] | None,
        correlation_id: str | None,
    ) -> Any:
        if self.events is None:
            return None
        return self.events.emit(
            EventType.EXTERNAL_WRITE,
            source=pid,
            target=resource,
            payload={
                "adapter": "shell",
                "operation": "run",
                "argv": argv,
                "returncode": proc.returncode,
                "policy_level": decision.policy_level,
                "high_risk": decision.high_risk,
                "risk": decision.risk.value,
                "rule_id": decision.rule_id,
                "cwd": os.fspath(cwd) if cwd is not None else None,
            },
            correlation_id=correlation_id,
            causality={"audit_parent_record_id": correlation_id} if correlation_id else {},
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

    def _profile_json(self, profile: SandboxProfile | None) -> dict[str, Any] | None:
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

    def _operation_context(
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
        profile_json = self._profile_json(profile)
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

    def _approval_constraints(
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

    def _selected_policy_level(self, caps: list[Capability]) -> str:
        return self._normalize_policy_level(caps[0].constraints[self.config.shell.policy_capability_key])

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

    def _first_matching_blacklist_rule(self, argv: list[str]) -> ShellCommandRule | None:
        direct = self._first_matching_rule(argv, self.config.shell.blacklist, allow_bare_only=False)
        if direct is not None:
            return direct
        # Some executables such as env/nohup/sudo can dispatch another program.
        # In blacklist mode, scan later tokens for exact executable identities
        # so `env bash -c ...` still requires human approval.
        executable_tokens = {self._normalize_executable(rule.argv[0]) for rule in self.config.shell.blacklist}
        for token in argv[1:]:
            if self._normalize_executable(token) in executable_tokens:
                return ShellCommandRule((token,), match="exact", description="nested blacklist executable")
        return None

    def _first_matching_rule(
        self,
        argv: list[str],
        rules: tuple[ShellCommandRule, ...],
        *,
        allow_bare_only: bool,
    ) -> ShellCommandRule | None:
        return next((rule for rule in rules if self._rule_matches(argv, rule, allow_bare_only=allow_bare_only)), None)

    def _rule_matches(self, argv: list[str], rule: ShellCommandRule, *, allow_bare_only: bool) -> bool:
        if not rule.argv:
            return False
        if allow_bare_only and self._argv0_has_path(argv[0]) and not self._argv0_has_path(rule.argv[0]):
            return False
        if rule.match == "exact" and len(argv) != len(rule.argv):
            return False
        if len(argv) < len(rule.argv):
            return False
        for index, expected in enumerate(rule.argv):
            actual = argv[index]
            if index == 0:
                if self._normalize_executable(actual) != self._normalize_executable(expected):
                    return False
                continue
            if actual != expected:
                return False
        return True

    def _validate_argv(self, argv: list[str]) -> list[str]:
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
        return checked

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

    def _provider_run(
        self,
        argv: list[str],
        *,
        timeout: float,
        cwd: str | os.PathLike[str] | None,
        limits: SubprocessLimits | None,
    ) -> CommandResult:
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
        return self.provider.run(argv, **kwargs)

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

    def _enforce_workspace_argv_scope(self, argv: list[str], *, cwd: str) -> None:
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
            for candidate in candidates:
                if self._is_path_like_argument(candidate, argv0=index == 0):
                    tokens.append(candidate)
        return tokens

    def _is_path_like_argument(self, value: str, *, argv0: bool) -> bool:
        if not value or self._looks_like_url(value):
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

    def _is_absolute_path(self, value: str) -> bool:
        return Path(value).is_absolute() or PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()

    def _resolve_argument_path(self, value: str, *, root: Path, cwd: Path) -> Path:
        if value.startswith("~"):
            raise CapabilityDenied(f"shell argument path uses host home expansion: {value}")
        raw = Path(value)
        if self._is_absolute_path(value):
            return raw.resolve()
        return (cwd / raw).resolve()
