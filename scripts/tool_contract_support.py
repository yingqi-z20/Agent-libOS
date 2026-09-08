"""Token-free tool contract inventory, canonical calls, and schema checks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Iterator

from jsonschema import Draft202012Validator

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.modules.core import register_module
from agent_libos.skills.builtin_catalog import BUILTIN_SKILL_MAX_INSTRUCTION_BYTES
from agent_libos.tools.base import BaseAgentTool
from agent_libos.utils.openai_schema import compact_model_json_schema


CANONICAL_CALLS: dict[str, dict[str, Any]] = {
    "create_checkpoint": {"reason": "Contract milestone"},
    "list_checkpoints": {},
    "fork_checkpoint": {"checkpoint_id": "checkpoint_observed"},
    "create_memory_object": {"type": "plan", "name": "ledger", "payload": {"entries": []}},
    "read_memory_object": {"name": "ledger"},
    "append_memory_object": {"name": "ledger", "entry": {"step": "verified"}},
    "list_memory_namespace": {},
    "create_memory_namespace": {"namespace": "project/child"},
    "create_object_from_file": {"name": "ledger", "path": "ledger.json"},
    "write_object_to_file": {"name": "ledger", "path": "ledger.json"},
    "process_exit": {"payload": {"summary": "Contract check"}},
    "write_text_file": {"path": "note.txt", "content": "Contract check\n"},
}

# Independent coverage expectations prevent removing one declaration from a
# multi-contract tool from silently reducing the checker/generator's scope.
REQUIRED_FIELD_CONTRACTS: dict[str, dict[str, str]] = {
    "create_checkpoint": {"pid": "current_process"},
    "list_checkpoints": {"pid": "current_process"},
    "fork_checkpoint": {"parent_pid": "detached_parent"},
    "create_memory_object": {"namespace": "current_namespace", "payload": "direct_json"},
    "read_memory_object": {"namespace": "current_namespace"},
    "append_memory_object": {"namespace": "current_namespace", "entry": "direct_json"},
    "list_memory_namespace": {"namespace": "current_namespace"},
    "create_memory_namespace": {"parent_namespace": "path_parent_namespace"},
    "create_object_from_file": {"namespace": "current_namespace"},
    "write_object_to_file": {"namespace": "current_namespace"},
    "process_exit": {"result_oid": "optional_result_object"},
    "write_text_file": {"expected_content_sha256": "content_precondition"},
}

REQUIRED_RESULT_FAMILIES = {
    "create_memory_object": "memory",
    "read_memory_object": "memory",
    "append_memory_object": "memory",
    "list_memory_namespace": "memory",
    "create_memory_namespace": "memory",
    "process_exit": "process_exit",
    "create_checkpoint": "checkpoint_creation",
}


def success_projection_cases() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Independent replay oracles; identifiers inside business data are literal."""

    current = "tenant:pid_contract_caller"
    business = {"ok": False, "namespace": current, "result_oid": "customer-value"}
    review = {"review_token": "review_observed", "requirements": [], "available_evidence_tools": []}
    cases: list[tuple[dict[str, Any], dict[str, Any]]] = [
        ({"tool_name": "create_memory_object", "result": {
            "oid": "obj_observed", "name": "ledger", "namespace": current, "type": "plan", "version": 2,
        }}, {"oid": "obj_observed", "name": "ledger", "namespace": None, "type": "plan"}),
        ({"tool_name": "append_memory_object", "result": {
            "name": "ledger", "namespace": current, "version": 2, "appended": True, "length": 1,
        }}, {"name": "ledger", "namespace": None, "appended": True, "length": 1}),
        ({"tool_name": "create_memory_namespace", "result": {
            "namespace": current + "/child", "parent_namespace": current, "created": True,
        }}, {"namespace": current + "/child", "parent_namespace": current, "created": True}),
        ({"tool_name": "list_memory_namespace", "result": {
            "namespace": current, "objects": [{"name": "ledger", "namespace": current}],
            "namespaces": [{"namespace": current + "/child", "parent_namespace": current}],
        }}, {"namespace": None, "objects": [{"name": "ledger", "namespace": None}],
             "namespaces": [{"namespace": current + "/child", "parent_namespace": current}]}),
        ({"tool_name": "process_exit", "result": {
            "status": "exited", "terminal_committed": True, "result_oid": "obj_host_only",
        }}, {"status": "exited", "terminal_committed": True}),
        ({"tool_name": "process_exit", "result": {
            "status": "completion_review_required", "completion_review": review,
            "terminal_committed": True, "result_oid": "obj_host_only",
        }}, {"status": "completion_review_required", "completion_review": review, "terminal_committed": False}),
        ({"tool_name": "process_exit", "ok": False, "model_projection": {
            "status": "exited", "terminal_committed": True,
            "error": {"code": "terminal_cleanup_required"},
        }}, {"status": "exited", "terminal_committed": True, "error": {"code": "terminal_cleanup_required"}}),
        ({"tool_name": "create_checkpoint", "result": {"checkpoint_id": "cp_host_only"},
          "model_projection": {"created": True, "reason": "Verified milestone"}},
         {"created": True, "reason": "Verified milestone"}),
        ({"tool_name": "read_memory_object", "result": {
            "name": "nullable", "representation": "json_value", "payload_type": "null", "payload": None,
        }}, {"name": "nullable", "representation": "json_value", "payload_type": "null", "payload": None}),
        ({"tool_name": "read_memory_object", "result": {
            "name": "paged", "representation": "canonical_json_page", "payload": None, "preview": '{"entries":',
        }}, {"name": "paged", "representation": "canonical_json_page", "preview": '{"entries":'}),
    ]
    for namespace, expected in ((current, None), ("process:self", "process:self"), ("tenant:foreign", "tenant:foreign")):
        cases.append((
            {"tool_name": "read_memory_object", "result": {"namespace": namespace, "name": "ledger", "payload": business}},
            {"namespace": expected, "name": "ledger", "payload": business},
        ))
    return cases


class _ToolInventory:
    def __init__(self) -> None:
        self.runtime = SimpleNamespace(config=DEFAULT_CONFIG)
        self.tools: list[BaseAgentTool] = []

    def register_tool(self, tool: BaseAgentTool) -> None:
        self.tools.append(tool)

    def register_image(self, image: Any) -> None:
        # Registration is observed, never executed against a Runtime.
        pass


def builtin_tools() -> list[BaseAgentTool]:
    """Collect precisely the core module's registrations without booting it."""

    inventory = _ToolInventory()
    register_module(inventory)  # type: ignore[arg-type]
    names = [tool.name for tool in inventory.tools]
    if len(names) != len(set(names)):
        raise ValueError("duplicate core tool names")
    return sorted(inventory.tools, key=lambda tool: tool.name)


def contract_configs() -> tuple[tuple[str, AgentLibOSConfig], ...]:
    """Exercise native defaults and a Host override, without provider access."""

    raised = replace(DEFAULT_CONFIG, skills=replace(
        DEFAULT_CONFIG.skills,
        package_max_bytes=BUILTIN_SKILL_MAX_INSTRUCTION_BYTES,
        resource_read_max_bytes=BUILTIN_SKILL_MAX_INSTRUCTION_BYTES // 2,
    ), tools=replace(
        DEFAULT_CONFIG.tools,
        max_sleep_seconds=DEFAULT_CONFIG.tools.max_sleep_seconds * 2,
        filesystem_read_hard_limit_bytes=DEFAULT_CONFIG.tools.filesystem_read_hard_limit_bytes * 2,
        directory_entry_hard_limit=DEFAULT_CONFIG.tools.directory_entry_hard_limit * 2,
    ))
    return (("default", DEFAULT_CONFIG), ("raised_tool_bounds_small_skills", raised))


def schema_nodes(schema: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Walk schema positions, never treating examples or property names as schemas."""

    pending = [schema]
    while pending:
        node = pending.pop()
        yield node
        for key in ("$defs", "definitions", "properties", "patternProperties", "dependentSchemas"):
            mapping = node.get(key)
            if isinstance(mapping, dict):
                pending.extend(value for value in mapping.values() if isinstance(value, dict))
        for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
            children = node.get(key)
            if isinstance(children, list):
                pending.extend(value for value in children if isinstance(value, dict))
        for key in ("items", "additionalProperties", "contains", "not", "if", "then", "else"):
            child = node.get(key)
            if isinstance(child, dict):
                pending.append(child)


def assert_wire_schema(source: dict[str, Any], wire: dict[str, Any], strict: bool) -> None:
    """Allow only documented strict-mode edits; reject partial fallback edits."""

    if type(strict) is not bool:
        raise ValueError("strict must be an explicit boolean")
    Draft202012Validator.check_schema(source)
    Draft202012Validator.check_schema(wire)
    expected = compact_model_json_schema(source)
    if strict:
        for node in schema_nodes(expected):
            if not (node.get("type") == "object" or "properties" in node or "additionalProperties" in node):
                continue
            if node.get("additionalProperties") is True or isinstance(node.get("additionalProperties"), dict):
                raise ValueError("strict conversion closes an intentionally open object")
            properties = node.setdefault("properties", {})
            node["required"] = list(properties)
            node["additionalProperties"] = False
    if wire != expected:
        raise ValueError("wire schema changed constraints or retained partial strict conversion edits")


def validate_field_value(schema: dict[str, Any], field: str, value: Any) -> bool:
    """Validate a single property with the enclosing reference definitions."""

    selected = {
        "$defs": deepcopy(schema.get("$defs", {})),
        "type": "object",
        "properties": {field: deepcopy(schema["properties"][field])},
        "required": [field],
        "additionalProperties": False,
    }
    return Draft202012Validator(selected).is_valid({field: value})
