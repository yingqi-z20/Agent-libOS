from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VALID_DOMAINS = {"coding", "research", "enterprise", "devops", "self_evolution"}
VALID_TRACKS = VALID_DOMAINS
VALID_VARIANTS = {"benign", "attack", "adaptive", "long_horizon"}
VALID_EFFECT_TYPES = {
    "filesystem.read",
    "filesystem.write",
    "filesystem.delete",
    "shell.exec",
    "object.read",
    "object.write",
    "human.request",
    "process.spawn",
    "process.fork",
    "process.exec",
    "skill.activate",
    "jit.register",
    "image.commit",
    "image.register",
    "checkpoint.create",
    "checkpoint.fork",
    "jsonrpc.call",
    "mcp.call",
    "email.send",
    "calendar.update",
    "crm.update",
    "ticket.update",
    "order.update",
    "external.network",
    "external.provider_call",
}


class PracticalValidationError(ValueError):
    pass


@dataclass(frozen=True)
class PracticalScenario:
    id: str
    title: str
    domain: str
    track: str
    task_family: str
    workflow: str
    variant: str
    attack_type: str
    goal: str
    workspace: str
    trusted_sources: list[str]
    untrusted_sources: list[str]
    allowed_effects: list[dict[str, Any]]
    forbidden_effects: list[dict[str, Any]]
    quality_oracle: list[dict[str, Any]]
    attack_oracle: list[dict[str, Any]]
    expected_provenance: list[dict[str, Any]]
    deterministic_actions: list[dict[str, Any]]
    utility_oracle: list[dict[str, Any]] = field(default_factory=list)
    security_oracle: list[dict[str, Any]] = field(default_factory=list)
    state_diff_oracle: list[dict[str, Any]] = field(default_factory=list)
    runtime_calls: list[dict[str, Any]] = field(default_factory=list)
    evidence_level: str = "modeled"
    service_state: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None
    source_path: Path | None = None


@dataclass
class PracticalEffect:
    scenario_id: str
    runner: str
    type: str
    requested: bool
    performed: bool
    denied: bool = False
    classification: str | None = None
    sensitive: bool = True
    path: str | None = None
    argv: list[str] | None = None
    endpoint: str | None = None
    method: str | None = None
    provider: str | None = None
    operation: str | None = None
    target: str | None = None
    actor: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "runner": self.runner,
            "type": self.type,
            "requested": self.requested,
            "performed": self.performed,
            "denied": self.denied,
            "classification": self.classification,
            "sensitive": self.sensitive,
            "path": self.path,
            "argv": self.argv,
            "endpoint": self.endpoint,
            "method": self.method,
            "provider": self.provider,
            "operation": self.operation,
            "target": self.target,
            "actor": self.actor,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class PracticalAuditRecord:
    scenario_id: str
    runner: str
    actor: str
    action: str
    target: str
    decision: dict[str, Any]
    effect_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "runner": self.runner,
            "actor": self.actor,
            "action": self.action,
            "target": self.target,
            "decision": self.decision,
            "effect_index": self.effect_index,
            "metadata": self.metadata,
        }


@dataclass
class PracticalResult:
    scenario_id: str
    runner: str
    domain: str
    variant: str
    ok: bool
    benign_success: bool
    attack_success_blocked: bool
    task_success: bool
    state_diff_success: bool
    evidence_level: str
    forbidden_effects: int
    forbidden_committed: int
    unknown_effects: int
    false_denials: int
    human_approvals: int
    tool_calls: int
    llm_tokens: int
    wall_time_s: float
    trace_coverage: float
    denial_explanation_coverage: float
    audit_query_latency_ms: float
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "runner": self.runner,
            "domain": self.domain,
            "variant": self.variant,
            "ok": self.ok,
            "benign_success": self.benign_success,
            "attack_success_blocked": self.attack_success_blocked,
            "task_success": self.task_success,
            "state_diff_success": self.state_diff_success,
            "evidence_level": self.evidence_level,
            "forbidden_effects": self.forbidden_effects,
            "forbidden_committed": self.forbidden_committed,
            "unknown_effects": self.unknown_effects,
            "false_denials": self.false_denials,
            "human_approvals": self.human_approvals,
            "tool_calls": self.tool_calls,
            "llm_tokens": self.llm_tokens,
            "wall_time_s": self.wall_time_s,
            "trace_coverage": self.trace_coverage,
            "denial_explanation_coverage": self.denial_explanation_coverage,
            "audit_query_latency_ms": self.audit_query_latency_ms,
            "errors": self.errors,
            "metadata": self.metadata,
        }


@dataclass
class PracticalRun:
    result: PracticalResult
    effects: list[PracticalEffect]
    audit: list[PracticalAuditRecord]
    human_requests: list[dict[str, Any]]
    llm_calls: list[dict[str, Any]]
    replay_actions: list[dict[str, Any]]
    service_state_before: dict[str, Any]
    service_state_after: dict[str, Any]
    external_effects: list[dict[str, Any]] = field(default_factory=list)
