from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from agent_libos.models import (
    DataFlowDirection,
    DataSink,
    EventType,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.sdk import (
    AuthorityMode,
    ProtectedOperationContract,
    ProtectedOperationEvidence,
    ProtectedOperationInvocation,
    ProtectedOperationSDK,
    ProviderPhase,
    ResourcePolicy,
)
from agent_libos.semantic.external import (
    ProtectedSemanticAssessmentCall,
    ProtectedSemanticCallPort,
)


SEMANTIC_LLM_ASSESS_CONTRACT = ProtectedOperationContract(
    name="semantic.llm.assess",
    provider="llm",
    operation="complete",
    evidence_roles=("audit", "effect", "event"),
    resource_policy=ResourcePolicy.NONE,
    authority_mode=AuthorityMode.RUNTIME_INTERNAL,
    state_mutation=True,
    information_flow=True,
    data_flow_direction=DataFlowDirection.EGRESS,
    require_classifier=False,
    internal_reason=(
        "the Host-owned Shadow worker invokes a pinned classifier only after "
        "the originating process manifest and DataFlow policy are enforced"
    ),
)


_T = TypeVar("_T")


class SdkProtectedSemanticCallPort(ProtectedSemanticCallPort):
    """Execute classifier I/O through a payload-minimizing protected effect.

    The adapter records only frozen profile, schema, and projection digests.
    The redacted projection is supplied solely to DataFlow preflight and never
    copied into audit, event, or external-effect metadata.
    """

    def __init__(
        self,
        sdk: ProtectedOperationSDK,
        *,
        profile_identity_resolver: Callable[[str], str],
    ) -> None:
        if not isinstance(sdk, ProtectedOperationSDK):
            raise TypeError("semantic protected port requires ProtectedOperationSDK")
        if not callable(profile_identity_resolver):
            raise TypeError("semantic profile identity resolver must be callable")
        self._sdk = sdk
        self._profile_identity_resolver = profile_identity_resolver
        self._sdk.register_contract(SEMANTIC_LLM_ASSESS_CONTRACT)

    def invoke(
        self,
        call: ProtectedSemanticAssessmentCall,
        *,
        provider: Any,
        dispatch: Callable[[], _T],
    ) -> _T:
        if not isinstance(call, ProtectedSemanticAssessmentCall):
            raise TypeError("semantic protected call has an invalid type")
        if not callable(dispatch):
            raise TypeError("semantic protected dispatch must be callable")
        if call.operation != SEMANTIC_LLM_ASSESS_CONTRACT.name:
            raise ValidationError("semantic protected operation identity drift")
        expected_effect = (
            f"{SEMANTIC_LLM_ASSESS_CONTRACT.provider}."
            f"{SEMANTIC_LLM_ASSESS_CONTRACT.operation}"
        )
        if call.effect != expected_effect:
            raise ValidationError("semantic protected effect identity drift")
        self._require_frozen_sink(call)

        invocation = ProtectedOperationInvocation(
            pid=call.pid,
            actor=call.actor,
            target=call.sink.identity,
            canonical_args=call.canonical_args,
            observation=call.canonical_args,
            data_sink=call.sink,
            data_sink_revalidator=lambda: self._current_sink(call),
            data_flow_context=call.data_flow_context,
            data_flow_payload=call.egress_payload,
            data_flow_operation=call.operation,
            data_flow_request_release=False,
            data_flow_redact_source_refs_evidence=True,
            failure_evidence=lambda error, phase: self._failure_evidence(
                call,
                error,
                phase,
            ),
        )
        with self._sdk.start(
            SEMANTIC_LLM_ASSESS_CONTRACT.name,
            invocation,
            provider=provider,
        ) as protected:
            result = protected.call(
                ProviderPhase(
                    "provider_request",
                    state_mutation=True,
                    information_flow=True,
                ),
                dispatch,
            )
            return protected.complete(
                result,
                self._success_evidence(call),
                classification_override=ExternalEffectClassification(
                    rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
                    rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
                    state_mutation=True,
                    information_flow=True,
                    metadata={"outcome": "provider_completed"},
                ),
            )

    def _require_frozen_sink(self, call: ProtectedSemanticAssessmentCall) -> None:
        expected_identity = f"llm:{call.profile_id}"
        if call.sink.identity != expected_identity:
            raise ValidationError("semantic classifier Sink identity is not profile-bound")
        current_sha256 = self._profile_identity_resolver(call.profile_id)
        if current_sha256 != call.profile_identity_sha256:
            raise ValidationError("semantic classifier profile identity drift")
        if call.sink.identity_sha256 != call.profile_identity_sha256:
            raise ValidationError("semantic classifier Sink provenance drift")

    def _current_sink(self, call: ProtectedSemanticAssessmentCall) -> DataSink:
        return DataSink(
            identity=f"llm:{call.profile_id}",
            identity_sha256=self._profile_identity_resolver(call.profile_id),
        )

    @staticmethod
    def _success_evidence(
        call: ProtectedSemanticAssessmentCall,
    ) -> ProtectedOperationEvidence:
        decision = {
            **call.canonical_args,
            "outcome": "provider_completed",
        }
        return ProtectedOperationEvidence(
            event_type=EventType.EXTERNAL_WRITE,
            event_source="runtime.semantic",
            event_target=call.sink.identity,
            event_payload=decision,
            audit_action="semantic.llm.assess",
            audit_actor=call.actor,
            audit_target=call.sink.identity,
            audit_decision=decision,
            effect_metadata={"semantic_assessment": call.canonical_args},
        )

    @staticmethod
    def _failure_evidence(
        call: ProtectedSemanticAssessmentCall,
        error: BaseException,
        phase: str,
    ) -> ProtectedOperationEvidence:
        decision = {
            **call.canonical_args,
            "outcome": "unknown",
            "phase": str(phase)[:128],
            "error_type": (type(error).__name__ or "BaseException")[:128],
        }
        return ProtectedOperationEvidence(
            event_type=EventType.EXTERNAL_WRITE,
            event_source="runtime.semantic",
            event_target=call.sink.identity,
            event_payload=decision,
            audit_action="semantic.llm.assess.failed",
            audit_actor=call.actor,
            audit_target=call.sink.identity,
            audit_decision=decision,
            effect_metadata={"semantic_assessment": call.canonical_args},
        )


__all__ = [
    "SEMANTIC_LLM_ASSESS_CONTRACT",
    "SdkProtectedSemanticCallPort",
]
