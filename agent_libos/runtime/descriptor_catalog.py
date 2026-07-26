from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from agent_libos.models import DataIntegrity, integrity_rank
from agent_libos.human.descriptors import (
    PROTECTED_OPERATION_DESCRIPTORS as HUMAN_OPERATIONS,
)
from agent_libos.llm.descriptors import (
    PROTECTED_OPERATION_DESCRIPTORS as LLM_OPERATIONS,
)
from agent_libos.modules.descriptors import (
    PROTECTED_OPERATION_DESCRIPTORS as MODULE_OPERATIONS,
)
from agent_libos.primitives.descriptors import (
    PROTECTED_OPERATION_DESCRIPTORS as PRIMITIVE_OPERATIONS,
)
from agent_libos.sdk import ProtectedOperationContract, ProtectedOperationSDK


PROTECTED_OPERATION_DESCRIPTORS: tuple[ProtectedOperationContract, ...] = (
    *PRIMITIVE_OPERATIONS,
    *LLM_OPERATIONS,
    *HUMAN_OPERATIONS,
    *MODULE_OPERATIONS,
)


def validate_descriptor_catalog() -> None:
    names = [descriptor.name for descriptor in PROTECTED_OPERATION_DESCRIPTORS]
    if len(names) != len(set(names)):
        raise ValueError("duplicate protected-operation descriptor")


validate_descriptor_catalog()


def register_protected_operation_descriptors(
    sdk: ProtectedOperationSDK,
    *,
    minimum_integrity: Mapping[str, DataIntegrity | str] | None = None,
) -> frozenset[str]:
    descriptors = configured_protected_operation_descriptors(minimum_integrity)
    for descriptor in descriptors:
        sdk.register_contract(descriptor)
    return frozenset(descriptor.name for descriptor in descriptors)


def configured_protected_operation_descriptors(
    minimum_integrity: Mapping[str, DataIntegrity | str] | None = None,
) -> tuple[ProtectedOperationContract, ...]:
    selected = dict(minimum_integrity or {})
    known = {descriptor.name for descriptor in PROTECTED_OPERATION_DESCRIPTORS}
    unknown = sorted(set(selected).difference(known))
    if unknown:
        raise ValueError(
            "unknown protected operation integrity override: " + ", ".join(unknown)
        )
    configured: list[ProtectedOperationContract] = []
    for descriptor in PROTECTED_OPERATION_DESCRIPTORS:
        override = DataIntegrity(
            selected.get(descriptor.name, descriptor.minimum_egress_integrity)
        )
        if integrity_rank(override) < integrity_rank(
            descriptor.minimum_egress_integrity
        ):
            raise ValueError(
                "protected operation integrity override cannot weaken contract: "
                f"{descriptor.name}"
            )
        configured.append(replace(descriptor, minimum_egress_integrity=override))
    return tuple(configured)


__all__ = [
    "PROTECTED_OPERATION_DESCRIPTORS",
    "configured_protected_operation_descriptors",
    "register_protected_operation_descriptors",
]
