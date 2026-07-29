from __future__ import annotations

from contextvars import ContextVar
from dataclasses import replace
from types import MethodType
from typing import Any, Mapping, Sequence

from agent_libos.capability.evaluator import CapabilityEvaluator
from agent_libos.models import (
    Capability,
    CapabilityDecision,
    CapabilityEffect,
    DataFlowOutcome,
    DataIntegrity,
    integrity_rank,
)


BENCHMARK_ONLY_ADMISSION_ABLATIONS: dict[str, dict[str, str | bool]] = {
    "no_task_ceiling": {
        "benchmark_only": True,
        "removed_gate": "task_authority_provider_effect_ceiling",
        "isolation": "per_runtime_instance_method_override",
    },
    "no_sink_clearance": {
        "benchmark_only": True,
        "removed_gate": "data_flow_sink_sensitivity_identity_clearance",
        "isolation": "per_runtime_instance_method_override",
    },
}


class BenchmarkNoPrimitiveApprovalEvaluator(CapabilityEvaluator):
    """Benchmark-only evaluator that removes the primitive ``ask`` step.

    Missing authority and explicit deny rules still fail closed.  Only an
    otherwise matching capability whose final decision is ``ask`` is promoted
    to ``allow``.  This keeps the ablation about approval rather than silently
    turning it into a broad-authority grant.
    """

    def decide(
        self,
        *,
        subject: str,
        resource: str,
        requested_right: str,
        matches: Sequence[Capability],
        context: Mapping[str, Any] | None = None,
        issuer_chains: Mapping[str, Sequence[str]] | None = None,
    ) -> CapabilityDecision:
        decision = super().decide(
            subject=subject,
            resource=resource,
            requested_right=requested_right,
            matches=matches,
            context=context,
            issuer_chains=issuer_chains,
        )
        if decision.effect != CapabilityEffect.ASK:
            return decision
        selected = next(
            (
                capability
                for capability in matches
                if capability.cap_id == decision.selected_capability_id
            ),
            None,
        )
        return replace(
            decision,
            allowed=True,
            effect=CapabilityEffect.ALLOW,
            reason="benchmark ablation bypassed primitive human approval",
            consume_capability_id=(
                selected.cap_id
                if selected is not None and selected.uses_remaining is not None
                else None
            ),
            human_request_id=None,
        )


def install_agent_libos_ablation(runtime: Any, runner: str) -> None:
    """Install one explicitly scoped benchmark intervention on ``runtime``."""

    if runner == "no_primitive_approval":
        capability = runtime.capability
        capability.evaluator = BenchmarkNoPrimitiveApprovalEvaluator(
            capability.rule_codec
        )
        _install_no_shell_approval(runtime)
        _install_no_git_approval(runtime)
    elif runner == "no_fork_attenuation":
        _install_no_fork_attenuation(runtime)
    elif runner == "no_task_ceiling":
        _install_no_task_ceiling(runtime)
    elif runner == "no_sink_clearance":
        _install_no_sink_clearance(runtime)


def benchmark_only_ablation_metadata(runner: str) -> dict[str, str | bool] | None:
    """Return a detached, machine-readable label for unsafe benchmark bypasses."""

    selected = BENCHMARK_ONLY_ADMISSION_ABLATIONS.get(runner)
    return dict(selected) if selected is not None else None


def _install_no_task_ceiling(runtime: Any) -> None:
    """Remove only Task Authority's provider effect-class ceiling.

    Capability checks, Sink clearance, resource accounting, provider contracts,
    and effect evidence remain unchanged.  Replacing a bound method on the
    freshly constructed benchmark instance makes this intervention impossible
    to select through production configuration.
    """

    def assert_effect_without_task_ceiling(
        authority_manifests: Any,
        pid: str,
        effect_class: str,
    ) -> None:
        del effect_class
        manifest = authority_manifests.get_for_process(pid)
        if manifest is not None:
            # Preserve production hash/provenance validation in ``get`` and
            # expiry enforcement in ``_require_live``.  Only the final
            # permitted-effect pattern comparison is omitted.
            authority_manifests._require_live(manifest)  # noqa: SLF001

    runtime.authority_manifests.assert_effect = MethodType(
        assert_effect_without_task_ceiling,
        runtime.authority_manifests,
    )


def _install_no_sink_clearance(runtime: Any) -> None:
    """Remove only Sink sensitivity/identity clearance on this Runtime.

    The surrounding ``authorize_egress`` path still validates source snapshots,
    target versions, registry generations, minimum integrity, conditional
    one-shot release, and ordinary authority.  It also records an explicit
    counterfactual clearance error when the benchmark bypass changes an allow
    decision.  This is a private instance override, not a Runtime option or
    Host API.
    """

    original_clearance_error = runtime.data_flow._clearance_error  # noqa: SLF001
    original_record_decision = runtime.data_flow._record_decision  # noqa: SLF001
    counterfactual_error: ContextVar[str | None] = ContextVar(
        f"benchmark_sink_clearance_error_{id(runtime.data_flow)}",
        default=None,
    )

    def clearance_without_sink_gate(
        data_flow: Any,
        *args: Any,
        **kwargs: Any,
    ) -> str | None:
        del data_flow
        error = original_clearance_error(*args, **kwargs)
        counterfactual_error.set(error)
        if error is None:
            return None
        labels = args[1] if len(args) >= 2 else kwargs.get("labels")
        minimum_integrity = DataIntegrity(
            kwargs.get("minimum_integrity", DataIntegrity.UNTRUSTED)
        )
        if labels is not None and integrity_rank(labels.integrity) < integrity_rank(
            minimum_integrity
        ):
            # Integrity is an operation-contract floor, not the Sink
            # sensitivity/identity predicate under test.
            return error
        return None

    def record_decision_with_bypass_evidence(
        data_flow: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        del data_flow
        bypassed = counterfactual_error.get()
        counterfactual_error.set(None)
        if kwargs.get("outcome") is DataFlowOutcome.ALLOW and bypassed is not None:
            kwargs["reason"] = (
                "BENCHMARK-ONLY bypassed Sink sensitivity/identity clearance: "
                f"{bypassed}"
            )
        return original_record_decision(*args, **kwargs)

    runtime.data_flow._clearance_error = MethodType(  # noqa: SLF001
        clearance_without_sink_gate,
        runtime.data_flow,
    )
    runtime.data_flow._record_decision = MethodType(  # noqa: SLF001
        record_decision_with_bypass_evidence,
        runtime.data_flow,
    )


def _install_no_fork_attenuation(runtime: Any) -> None:
    """Make benchmark children receive an unattenuated copy of parent grants.

    This deliberately bypasses the production ``derive_authority`` path.  The
    intervention is installed only on the per-task benchmark Runtime instance;
    it is not a supported Runtime configuration or embedding-host API.
    """

    def compile_unattenuated_child_authority(
        process_manager: Any,
        *,
        parent_pid: str,
        child_pid: str,
        manifest: Any | None,
        requested_capabilities: list[dict[str, Any]],
        inherit_specs: list[dict[str, Any]],
        transition_kind: str,
    ) -> None:
        del process_manager, manifest, requested_capabilities, inherit_specs
        parent_capabilities = runtime.capability.capabilities_for(parent_pid)
        # ``list_subject`` intentionally applies the configured inspection
        # limit when no explicit limit is supplied.  This ablation promises to
        # remove attenuation for every active parent grant, so size the
        # authoritative active view to the complete persisted subject set.
        active_capability_ids = {
            capability.cap_id
            for capability in runtime.capability.list_subject(
                parent_pid,
                limit=max(1, len(parent_capabilities)),
            )
        }
        for capability in parent_capabilities:
            if capability.cap_id not in active_capability_ids:
                continue
            runtime.capability.issue_trusted(
                subject=child_pid,
                resource=capability.resource,
                rights=capability.rights,
                issued_by=f"benchmark:no_fork_attenuation:{parent_pid}",
                effect=capability.effect,
                constraints=capability.constraints,
                metadata={
                    **capability.metadata,
                    "benchmark_ablation": "no_fork_attenuation",
                    "benchmark_source_capability_id": capability.cap_id,
                    "benchmark_transition_kind": transition_kind,
                },
                expires_at=capability.expires_at,
                uses_remaining=capability.uses_remaining,
                delegable=capability.delegable,
                revocable=capability.revocable,
                max_delegation_depth=capability.max_delegation_depth,
            )

    runtime.process._compile_child_authority = MethodType(  # noqa: SLF001
        compile_unattenuated_child_authority,
        runtime.process,
    )


def _install_no_shell_approval(runtime: Any) -> None:
    original_authorize = runtime.shell.authorize_operation

    def authorize_without_approval(
        shell_adapter: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        del shell_adapter
        decision = original_authorize(*args, **kwargs)
        authority = decision.authority_decision
        if not decision.ask_human:
            return decision
        # An ASK caused only by a shell classification rule may sit on top of
        # an already-allowed policy capability.  Missing or denied capability
        # authority is never promoted by this intervention.
        if authority is None or not authority.allowed:
            return decision
        return replace(
            decision,
            allowed=True,
            ask_human=False,
            reason="benchmark ablation bypassed shell human approval",
        )

    runtime.shell.authorize_operation = MethodType(
        authorize_without_approval,
        runtime.shell,
    )


def _install_no_git_approval(runtime: Any) -> None:
    original_authorize = runtime.git._authorize  # noqa: SLF001
    original_authorize_mutation = runtime.git._authorize_mutation  # noqa: SLF001

    def authorize_without_mandatory_approval(
        git_primitive: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        del git_primitive
        kwargs["mandatory_approval"] = False
        return original_authorize(*args, **kwargs)

    def authorize_mutation_without_approval_binding(
        git_primitive: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        del git_primitive
        decisions, _, _, filesystem_scopes = original_authorize_mutation(
            *args,
            **kwargs,
        )
        return decisions, set(), None, filesystem_scopes

    runtime.git._authorize = MethodType(  # noqa: SLF001
        authorize_without_mandatory_approval,
        runtime.git,
    )
    runtime.git._authorize_mutation = MethodType(  # noqa: SLF001
        authorize_mutation_without_approval_binding,
        runtime.git,
    )


_SANDBOX_ONLY_ALLOWED_ACTIONS = frozenset(
    {
        "process_exit",
        "read_text_file",
        "write_text_file",
        "delete_file",
        "delete_directory",
        "read_memory_object",
        "create_memory_object",
        "append_memory_object",
    }
)


def sandbox_only_denial_reason(action: Mapping[str, Any]) -> str | None:
    """Return the static tool-category denial for the sandbox-only baseline.

    The sandbox baseline exposes only fixture-contained filesystem operations
    and its in-memory object map.  It has no shell, provider, process-control,
    Skill, JIT, image, checkpoint, or remote execution surface.  This makes it
    a real intervention distinct from the direct wrapper while remaining
    deterministic and token-free.
    """

    action_name = str(action.get("action") or "")
    if action_name in _SANDBOX_ONLY_ALLOWED_ACTIONS:
        return None
    return f"sandbox-only baseline blocks action category: {action_name or '<missing>'}"


__all__ = [
    "BENCHMARK_ONLY_ADMISSION_ABLATIONS",
    "BenchmarkNoPrimitiveApprovalEvaluator",
    "benchmark_only_ablation_metadata",
    "install_agent_libos_ablation",
    "sandbox_only_denial_reason",
]
