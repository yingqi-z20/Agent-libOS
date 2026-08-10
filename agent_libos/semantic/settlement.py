from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from typing import Any

from agent_libos.models import HumanRequest, HumanRequestStatus
from agent_libos.models.exceptions import ValidationError
from agent_libos.models.semantic import DeterministicDenyDecision
from agent_libos.models.semantic import SemanticRuntimeMode
from agent_libos.semantic.enforcement import HostSemanticControlFence


class HostSemanticDenySettlement:
    """Atomically bind a deterministic deny to semantic and Human evidence."""

    def __init__(
        self,
        *,
        transaction: Callable[[], AbstractContextManager[Any]],
        machine_terminalizer: Callable[..., tuple[HumanRequest, dict[str, Any]]],
        control_fence: HostSemanticControlFence | None = None,
    ) -> None:
        if not callable(transaction) or not callable(machine_terminalizer):
            raise TypeError("semantic deny settlement dependencies must be callable")
        self._transaction = transaction
        self._machine_terminalizer = machine_terminalizer
        if control_fence is not None and not isinstance(
            control_fence,
            HostSemanticControlFence,
        ):
            raise TypeError("semantic deny control fence is invalid")
        self._control_fence = control_fence

    def settle_deny(
        self,
        *,
        request_id: str,
        expected_revision: int,
        decision: Any,
        semantic_terminalizer: Callable[[], bool],
    ) -> tuple[HumanRequest, dict[str, Any]]:
        if not isinstance(decision, DeterministicDenyDecision):
            raise TypeError("semantic deny settlement requires a deterministic proof")
        if (
            decision.request_id != request_id
            or decision.request_revision != expected_revision
        ):
            raise ValidationError(
                "semantic deny proof does not match the requested Human CAS"
            )
        if not callable(semantic_terminalizer):
            raise TypeError("semantic deny terminalizer must be callable")
        with self._transaction():
            if self._control_fence is None:
                raise ValidationError(
                    "semantic deterministic deny has no control commit fence"
                )
            self._control_fence.fence(
                expected_policy_sha256=decision.policy_sha256,
                allowed_modes=(
                    SemanticRuntimeMode.ENFORCE_DENY,
                    SemanticRuntimeMode.CANARY_AUTO,
                ),
            )
            settled, evidence = self._machine_terminalizer(
                request_id,
                expected_revision=expected_revision,
                status=HumanRequestStatus.REJECTED,
                decision={
                    "schema_version": 1,
                    "deterministic_deny_sha256": decision.canonical_sha256(),
                    "reason_codes": [
                        reason.value for reason in decision.reason_codes
                    ],
                    "policy_sha256": decision.policy_sha256,
                },
                responder="policy:semantic:hard-deny",
                authority_applier=None,
                audit_action="semantic.policy.deny",
            )
            if not isinstance(evidence, Mapping):
                raise ValidationError(
                    "semantic machine deny returned invalid evidence"
                )
            if semantic_terminalizer() is not True:
                raise ValidationError(
                    "semantic job/assessment deny CAS was not committed"
                )
            return settled, dict(evidence)


__all__ = ["HostSemanticDenySettlement"]
