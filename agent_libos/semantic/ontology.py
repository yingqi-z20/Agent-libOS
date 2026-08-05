from __future__ import annotations

from dataclasses import dataclass

from agent_libos.models import AuthorityRisk
from agent_libos.models.semantic import SemanticApprovalRule, SemanticDomain


@dataclass(frozen=True, slots=True)
class SemanticActionDefinition:
    action_id: str
    domain: SemanticDomain
    authority_operation: str
    allowed_rights: tuple[str, ...]
    risk: AuthorityRisk
    auto_approval_eligible: bool
    requires_data_flow_egress: bool


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
            if type(action.requires_data_flow_egress) is not bool:
                raise TypeError(
                    "semantic action egress requirement must be a boolean"
                )
            if action.authority_operation in by_operation:
                raise ValueError(f"duplicate semantic authority operation: {action.authority_operation}")
            by_operation[action.authority_operation] = action
        self._by_operation = by_operation

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
