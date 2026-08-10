from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from agent_libos.models.capability import AuthorityRisk
from agent_libos.models.semantic import (
    SEMANTIC_ACTION_CATALOG_V1,
    SEMANTIC_ACTION_CATALOG_VERSION,
    SemanticApprovalRule,
    SemanticDomain,
)


_KNOWN_RIGHTS = frozenset(
    {"read", "write", "execute", "link", "diff", "materialize", "delete"}
)


@dataclass(frozen=True, slots=True)
class SemanticActionDefinition:
    action_id: str
    domain: SemanticDomain
    authority_operation: str
    allowed_rights: tuple[str, ...]
    risk: AuthorityRisk
    auto_approval_eligible: bool
    requires_data_flow_egress: bool

    def __post_init__(self) -> None:
        _validate_action_identity(self)
        _validate_action_rights(self)
        _normalize_action_risk(self)
        _validate_action_flags(self)
        _validate_action_catalog_entry(self)


def _validate_action_identity(action: SemanticActionDefinition) -> None:
    if (
        type(action.action_id) is not str
        or action.action_id != action.authority_operation
        or "." not in action.action_id
        or "*" in action.action_id
    ):
        raise ValueError(
            "semantic action identity must equal its exact authority operation"
        )
    if not isinstance(action.domain, SemanticDomain):
        object.__setattr__(action, "domain", SemanticDomain(action.domain))
    if action.action_id.split(".", 1)[0] != action.domain.value:
        raise ValueError("semantic action domain must match its operation prefix")


def _validate_action_rights(action: SemanticActionDefinition) -> None:
    if not isinstance(action.allowed_rights, tuple) or not action.allowed_rights:
        raise TypeError("semantic action allowed_rights must be a non-empty tuple")
    if len(action.allowed_rights) != len(set(action.allowed_rights)) or any(
        type(right) is not str or right not in _KNOWN_RIGHTS
        for right in action.allowed_rights
    ):
        raise ValueError("semantic action rights must be unique non-control rights")


def _normalize_action_risk(action: SemanticActionDefinition) -> None:
    if not isinstance(action.risk, AuthorityRisk):
        object.__setattr__(action, "risk", AuthorityRisk(action.risk))


def _validate_action_flags(action: SemanticActionDefinition) -> None:
    if type(action.auto_approval_eligible) is not bool:
        raise TypeError("semantic action auto eligibility must be a boolean")
    if type(action.requires_data_flow_egress) is not bool:
        raise TypeError("semantic action egress requirement must be a boolean")


def _validate_action_catalog_entry(action: SemanticActionDefinition) -> None:
    frozen_rights = SEMANTIC_ACTION_CATALOG_V1.get(action.authority_operation)
    if action.auto_approval_eligible:
        if (
            frozen_rights is None
            or frozenset(action.allowed_rights) != frozen_rights
            or action.risk is not AuthorityRisk.LOW
            or action.requires_data_flow_egress
        ):
            raise ValueError(
                "semantic auto-eligible action must exactly match catalog v1"
            )
    elif frozen_rights is not None:
        raise ValueError(
            "semantic catalog v1 auto action cannot be reclassified as ineligible"
        )


_DEFAULT_ACTIONS = (
    SemanticActionDefinition("filesystem.read", SemanticDomain.FILESYSTEM, "filesystem.read", ("read",), AuthorityRisk.LOW, True, False),
    SemanticActionDefinition("filesystem.diff", SemanticDomain.FILESYSTEM, "filesystem.diff", ("diff",), AuthorityRisk.LOW, False, False),
    SemanticActionDefinition("filesystem.write", SemanticDomain.FILESYSTEM, "filesystem.write", ("write",), AuthorityRisk.HIGH, False, True),
    SemanticActionDefinition("filesystem.delete", SemanticDomain.FILESYSTEM, "filesystem.delete", ("delete",), AuthorityRisk.DESTRUCTIVE, False, True),
    SemanticActionDefinition("filesystem.execute", SemanticDomain.FILESYSTEM, "filesystem.execute", ("execute",), AuthorityRisk.HIGH, False, True),
    SemanticActionDefinition("filesystem.link", SemanticDomain.FILESYSTEM, "filesystem.link", ("link",), AuthorityRisk.HIGH, False, True),
    SemanticActionDefinition("filesystem.materialize", SemanticDomain.FILESYSTEM, "filesystem.materialize", ("materialize",), AuthorityRisk.MEDIUM, False, True),
    SemanticActionDefinition("git.read", SemanticDomain.GIT, "git.read", ("read",), AuthorityRisk.LOW, True, False),
    SemanticActionDefinition("git.diff", SemanticDomain.GIT, "git.diff", ("diff",), AuthorityRisk.LOW, True, False),
    SemanticActionDefinition("git.write", SemanticDomain.GIT, "git.write", ("write",), AuthorityRisk.HIGH, False, True),
    SemanticActionDefinition("git.execute", SemanticDomain.GIT, "git.execute", ("execute",), AuthorityRisk.HIGH, False, True),
    SemanticActionDefinition("shell.run", SemanticDomain.SHELL, "shell.run", ("execute",), AuthorityRisk.HIGH, False, True),
    SemanticActionDefinition("jsonrpc.call", SemanticDomain.JSONRPC, "jsonrpc.call", ("read", "write", "execute"), AuthorityRisk.HIGH, False, True),
    SemanticActionDefinition("mcp.call", SemanticDomain.MCP, "mcp.call", ("read", "write", "execute"), AuthorityRisk.HIGH, False, True),
)


class ActionOntology:
    """Immutable Host-owned operation registry used by the Shadow broker."""

    def __init__(self, actions: tuple[SemanticActionDefinition, ...] = _DEFAULT_ACTIONS):
        if not isinstance(actions, tuple) or not actions:
            raise ValueError("semantic action ontology must contain actions")
        by_operation: dict[str, SemanticActionDefinition] = {}
        for action in actions:
            if not isinstance(action, SemanticActionDefinition):
                raise TypeError("semantic action ontology entries must be SemanticActionDefinition")
            if action.authority_operation in by_operation:
                raise ValueError(f"duplicate semantic authority operation: {action.authority_operation}")
            by_operation[action.authority_operation] = action
        auto_operations = {
            operation
            for operation, action in by_operation.items()
            if action.auto_approval_eligible
        }
        if auto_operations != set(SEMANTIC_ACTION_CATALOG_V1):
            raise ValueError(
                "semantic action ontology must contain exactly catalog v1 auto operations"
            )
        self._by_operation = MappingProxyType(by_operation)

    @property
    def catalog_version(self) -> int:
        return SEMANTIC_ACTION_CATALOG_VERSION

    def is_catalog_v1_auto_operation(self, authority_operation: str) -> bool:
        return authority_operation in SEMANTIC_ACTION_CATALOG_V1

    def resolve(self, authority_operation: str) -> SemanticActionDefinition | None:
        if type(authority_operation) is not str:
            return None
        return self._by_operation.get(authority_operation)

    def validate_auto_rule(self, rule: SemanticApprovalRule) -> SemanticActionDefinition:
        action = self.resolve(rule.authority_operation)
        if action is None:
            raise ValueError("semantic rule references an unknown authority operation")
        if not action.auto_approval_eligible:
            raise ValueError("semantic rule references an operation that is structurally ineligible")
        if not set(rule.rights).issubset(action.allowed_rights):
            raise ValueError("semantic rule rights exceed the action ontology")
        return action

    def definitions(self) -> tuple[SemanticActionDefinition, ...]:
        return tuple(self._by_operation[key] for key in sorted(self._by_operation))


DEFAULT_ACTION_ONTOLOGY = ActionOntology()


__all__ = ["ActionOntology", "DEFAULT_ACTION_ONTOLOGY", "SemanticActionDefinition"]
