from agent_libos.models import DataFlowDirection
from agent_libos.sdk import AuthorityMode, ResourcePolicy
from agent_libos.sdk.descriptors import protected_operation_descriptor as operation


PROTECTED_OPERATION_DESCRIPTORS = (
    operation(
        "primitive.llm.complete",
        "llm",
        "complete",
        resource_policy=ResourcePolicy.REQUIRED,
        authority_mode=AuthorityMode.RUNTIME_INTERNAL,
        state_mutation=True,
        information_flow=True,
        data_flow_direction=DataFlowDirection.BIDIRECTIONAL,
        require_classifier=False,
        internal_reason=(
            "the scheduler owns model selection after process authority and context "
            "materialization; DataFlowManager independently gates the provider Sink"
        ),
    ),
)


__all__ = ["PROTECTED_OPERATION_DESCRIPTORS"]
