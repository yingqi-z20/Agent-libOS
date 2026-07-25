from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    "external.network",
    "external.provider_call",
}

VALID_EFFECT_OUTCOMES = {"performed", "denied", "not_started", "simulated", "unknown"}
VALID_EFFECT_EVIDENCE = {
    "runtime_external_effect",
    "runtime_audit",
    "runtime_result_denial",
    "wrapper_observed",
    "benchmark_simulation",
    "missing",
}

BENCHMARK_EFFECT_OBSERVATION_FIELDS = {
    "effect_id",
    "performed",
    "denied",
    "simulated",
    "outcome",
    "evidence",
    "error",
    "classification",
    "metadata",
}


class BenchmarkValidationError(ValueError):
    pass


@dataclass(frozen=True)
class BenchmarkTask:
    id: str
    title: str
    goal: str
    workspace: str
    attack_class: str
    allowed_effects: list[dict[str, Any]]
    forbidden_effects: list[dict[str, Any]]
    success_oracle: list[dict[str, Any]]
    safety_oracle: list[dict[str, Any]]
    schema_version: int = 1
    setup: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    human_responses: list[dict[str, Any]] = field(default_factory=list)
    expected_audit: list[dict[str, Any]] = field(default_factory=list)
    mock_actions: list[dict[str, Any]] = field(default_factory=list)
    notes: str | None = None
    source_path: Path | None = None
    source_sha256: str | None = None


@dataclass
class EffectRecord:
    task_id: str
    runner: str
    type: str
    performed: bool
    denied: bool = False
    simulated: bool = False
    effect_id: str | None = None
    outcome: str | None = None
    evidence: str | None = None
    path: str | None = None
    argv: list[str] | None = None
    namespace: str | None = None
    name: str | None = None
    skill_id: str | None = None
    tool: str | None = None
    image: str | None = None
    checkpoint: str | None = None
    resource: str | None = None
    operation: str | None = None
    endpoint: str | None = None
    method: str | None = None
    provider: str | None = None
    error: str | None = None
    classification: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "effect_id": self.effect_id,
            "task_id": self.task_id,
            "runner": self.runner,
            "type": self.type,
            "performed": self.performed,
            "denied": self.denied,
            "simulated": self.simulated,
            "outcome": self.outcome,
            "evidence": self.evidence,
            "path": self.path,
            "argv": self.argv,
            "namespace": self.namespace,
            "name": self.name,
            "skill_id": self.skill_id,
            "tool": self.tool,
            "image": self.image,
            "checkpoint": self.checkpoint,
            "resource": self.resource,
            "operation": self.operation,
            "endpoint": self.endpoint,
            "method": self.method,
            "provider": self.provider,
            "error": self.error,
            "classification": self.classification,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkResult:
    task_id: str
    runner: str
    attack_class: str
    ok: bool
    task_success: bool
    safety_passed: bool
    unknown_effects: int
    forbidden_performed: int
    approval_count: int
    tool_calls: int
    primitive_calls: int
    llm_tokens: int
    wall_time_s: float
    audit_records: int
    audit_completeness: float
    valid: bool = True
    invalid_reasons: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    workspace: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "runner": self.runner,
            "attack_class": self.attack_class,
            "ok": self.ok,
            "task_success": self.task_success,
            "safety_passed": self.safety_passed,
            "unknown_effects": self.unknown_effects,
            "forbidden_performed": self.forbidden_performed,
            "approval_count": self.approval_count,
            "tool_calls": self.tool_calls,
            "primitive_calls": self.primitive_calls,
            "llm_tokens": self.llm_tokens,
            "wall_time_s": self.wall_time_s,
            "audit_records": self.audit_records,
            "audit_completeness": self.audit_completeness,
            "valid": self.valid,
            "invalid_reasons": self.invalid_reasons,
            "errors": self.errors,
            "workspace": self.workspace,
            "metadata": self.metadata,
        }


@dataclass
class TaskRun:
    result: BenchmarkResult
    effects: list[EffectRecord]
