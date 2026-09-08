"""Declarative model-facing tool contracts, independent of execution authority.

Field metadata stays local to Pydantic; provider schemas contain only the
generated description and ordinary JSON Schema constraints. Contracts describe
canonical calls, not coercions or capability grants. Compatibility validators
and the underlying primitives still decide what an actual call may do.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_core import PydanticUndefined

_NON_TEXT_VALUES = ('0', 'false', '[]', '{}')


@dataclass(frozen=True, slots=True)
class FieldContract:
    """One source for a field's guidance, default, and lexical constraints."""

    key: str
    description: str
    nullable: bool
    default_is_null: bool = False
    min_length: int | None = None
    pattern: str | None = None
    # JSON values encoded as text keep the contract itself immutable. The
    # checker tests exact type/value preservation, including literal strings.
    canonical_values: tuple[str, ...] = ()
    rejected_values: tuple[str, ...] = ()

    def field(self) -> Any:
        """Create fresh Pydantic metadata without custom wire-schema keywords."""

        constraints: dict[str, Any] = {}
        if self.min_length is not None:
            constraints["min_length"] = self.min_length
        if self.pattern is not None:
            constraints["pattern"] = self.pattern
        field = Field(
            default=None if self.default_is_null else PydanticUndefined,
            description=self.description,
            **constraints,
        )
        field.metadata.append(self)
        return field


CURRENT_PROCESS = FieldContract(
    key="current_process",
    description=(
        "Target process id. Pass JSON null to select the caller; omission is "
        "also valid when allowed by the call schema. Strings are exact process "
        "ids. Do not guess 'self' or infer a pid "
        "from a projected Capability resource."
    ),
    nullable=True,
    default_is_null=True,
    canonical_values=('null', '"pid_observed"', '"self"'),
    rejected_values=_NON_TEXT_VALUES,
)
CURRENT_NAMESPACE = FieldContract(
    key="current_namespace",
    description=(
        "Object Memory namespace. Pass JSON null to select this process "
        "namespace; omission is also valid when allowed by the call schema. "
        "Explicit strings name exact namespaces, including literal 'process:self'; "
        "they are not aliases. Do not broaden to a parent namespace after denial."
    ),
    nullable=True,
    default_is_null=True,
    canonical_values=('null', '"process:pid_observed"', '"process:self"'),
    rejected_values=_NON_TEXT_VALUES,
)
PATH_PARENT_NAMESPACE = FieldContract(
    key="path_parent_namespace",
    description=(
        "Parent namespace. JSON null selects the path parent; top-level "
        "namespaces have no parent. Omission is also valid when allowed by "
        "the call schema. An explicit string names the exact parent namespace."
    ),
    nullable=True,
    default_is_null=True,
    canonical_values=('null', '"project"'),
    rejected_values=_NON_TEXT_VALUES,
)
DETACHED_PARENT = FieldContract(
    key="detached_parent",
    description=(
        "Parent of the fork root. Pass JSON null for "
        "a detached root; omission is also valid when allowed by the call schema. "
        "Strings select exact parents. Null does not select the caller."
    ),
    nullable=True,
    default_is_null=True,
    canonical_values=('null', '"pid_observed"', '"self"'),
    rejected_values=_NON_TEXT_VALUES,
)
OPTIONAL_RESULT_OBJECT = FieldContract(
    key="optional_result_object",
    description=(
        "Existing non-empty object id to use as process result. When there is "
        "no existing result Object, pass JSON null; omission is also valid when "
        "allowed by the call schema. Never pass an empty string or the text "
        "'None' or 'null'."
    ),
    nullable=True,
    default_is_null=True,
    min_length=1,
    canonical_values=('null', '"obj_observed"'),
    rejected_values=(*_NON_TEXT_VALUES, '""'),
)
DIRECT_JSON = FieldContract(
    key="direct_json",
    description=(
        "Direct JSON value. JSON strings are stored literally; pass an object "
        "or array value, not a JSON-encoded string, when a container is intended."
    ),
    nullable=True,
    canonical_values=('null', '{"entries":[]}', '[]', '"{\\"entries\\":[]}"', '0', 'false'),
)
CONTENT_PRECONDITION = FieldContract(
    key="content_precondition",
    description=(
        "Compare-and-swap precondition. Pass the full-content SHA-256 returned "
        "by read_text_file, or 'missing' to require creation. JSON null disables "
        "this precondition; omission is also valid when allowed by the call schema. "
        "Do not replace a rejected precondition with null merely to make a write pass."
    ),
    nullable=True,
    default_is_null=True,
    pattern=r"^(?:missing|[0-9a-f]{64})$",
    canonical_values=('null', '"missing"', '"' + 'a' * 64 + '"'),
    rejected_values=(*_NON_TEXT_VALUES, '""', '"missing "', '"abc"', '"' + 'A' * 64 + '"'),
)

FIELD_CONTRACTS = MappingProxyType({
    contract.key: contract for contract in (
        CURRENT_PROCESS, CURRENT_NAMESPACE, PATH_PARENT_NAMESPACE,
        DETACHED_PARENT, OPTIONAL_RESULT_OBJECT, DIRECT_JSON, CONTENT_PRECONDITION,
    )
})


def declared_field_contracts(model: type[BaseModel]) -> dict[str, FieldContract]:
    """Discover declarations from actual input fields, including inherited ones."""

    selected: dict[str, FieldContract] = {}
    for name, field in model.model_fields.items():
        contracts = [item for item in field.metadata if isinstance(item, FieldContract)]
        if len(contracts) > 1:
            raise ValueError(f"{model.__name__}.{name}: multiple field contracts")
        if contracts:
            selected[name] = contracts[0]
    return selected


@dataclass(frozen=True, slots=True)
class ResultContract:
    family: Literal["memory", "process_exit", "checkpoint_creation"]
    preserve_public_failure: bool
    guidance: str


_MEMORY_RESULT = ResultContract(
    family="memory", preserve_public_failure=True,
    guidance=(
        "A failed call remains ok=false with a safe error type and recovery hint when available; "
        "a missing result is not success. A complete json_value read preserves payload even "
        "when it is JSON null; a canonical_json_page read carries a preview instead."
    ),
)
_EXIT_RESULT = ResultContract(
    family="process_exit", preserve_public_failure=True,
    guidance="Only status=exited with terminal_committed=true confirms exit; completion_review_required is nonterminal. Failed calls retain safe diagnostics.",
)
CHECKPOINT_CREATION = ResultContract(
    family="checkpoint_creation", preserve_public_failure=False,
    guidance=(
        "In v2, a successful tool result with `created: true` confirms creation without Host ids. "
        "Do not create another checkpoint to obtain an id; list only when a later "
        "operation needs selection. Legacy results retain `checkpoint_id`."
    ),
)
RESULT_CONTRACTS = MappingProxyType({
    **{name: _MEMORY_RESULT for name in (
        "create_memory_object", "create_memory_namespace", "list_memory_namespace",
        "read_memory_object", "append_memory_object",
    )},
    "process_exit": _EXIT_RESULT,
    "create_checkpoint": CHECKPOINT_CREATION,
})


def compact_checkpoint_created(reason: str) -> dict[str, Any]:
    """Return the declared successful v2 receipt; never invent an identity."""

    return {"created": True, "reason": reason}
