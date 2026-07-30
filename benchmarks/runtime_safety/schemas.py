from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from benchmarks.runtime_safety.models import (
    VALID_EFFECT_EVIDENCE,
    VALID_EFFECT_OUTCOMES,
    VALID_EFFECT_TYPES,
)


JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
EXPECTED_EFFECT_OUTCOMES = frozenset({"performed", "denied", "simulated"})
POLICY_DEFAULTS: dict[str, Any] = {
    "confirmation_default_decision": "approve",
    "human_auto_approve": False,
    "human_auto_policy": None,
    "human_auto_answer": None,
}

_NON_EMPTY_STRING = {"type": "string", "minLength": 1, "pattern": r"\S"}
_NULLABLE_STRING = {"type": ["string", "null"]}
_BOOLEAN = {"type": "boolean"}
_POSITIVE_INTEGER = {"type": "integer", "minimum": 1}
_NON_NEGATIVE_INTEGER = {"type": "integer", "minimum": 0}
_STRING_LIST = {"type": "array", "items": {"type": "string"}}
_NON_EMPTY_STRING_LIST = {
    "type": "array",
    "minItems": 1,
    "items": deepcopy(_NON_EMPTY_STRING),
}
_GOAL = {
    "oneOf": [
        deepcopy(_NON_EMPTY_STRING),
        {"type": "object"},
    ]
}
_STATE_TOKEN = {
    "oneOf": [
        {"const": "$git_state_token"},
        {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    ]
}
_GIT_PATH = {
    "oneOf": [
        deepcopy(_NON_EMPTY_STRING),
        {
            "type": "object",
            "properties": {
                "path": deepcopy(_NULLABLE_STRING),
                "path_b64": deepcopy(_NULLABLE_STRING),
            },
            "additionalProperties": False,
            "oneOf": [
                {"required": ["path"], "properties": {"path": deepcopy(_NON_EMPTY_STRING)}},
                {"required": ["path_b64"], "properties": {"path_b64": deepcopy(_NON_EMPTY_STRING)}},
            ],
        },
    ]
}


def _exact_benchmark_effect(
    effect_type: str,
    **identity: str,
) -> dict[str, Any]:
    properties: dict[str, Any] = {"type": {"const": effect_type}}
    properties.update({key: {"const": value} for key, value in identity.items()})
    return {
        "type": "array",
        "minItems": 1,
        "maxItems": 1,
        "prefixItems": [
            {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            }
        ],
        "items": False,
    }


def _action_contract(
    name: str,
    *,
    required: dict[str, dict[str, Any]] | None = None,
    optional: dict[str, dict[str, Any]] | None = None,
    benchmark_effects: dict[str, Any] | None = None,
    require_benchmark_effects: bool = False,
) -> dict[str, Any]:
    required = required or {}
    optional = optional or {}
    properties: dict[str, Any] = {
        "action": {"const": name},
        "process_goal": deepcopy(_NON_EMPTY_STRING),
        **deepcopy(required),
        **deepcopy(optional),
    }
    required_names = ["action", *required]
    if benchmark_effects is not None:
        properties["benchmark_effects"] = deepcopy(benchmark_effects)
        if require_benchmark_effects:
            required_names.append("benchmark_effects")
    return {
        "type": "object",
        "properties": properties,
        "required": required_names,
        "additionalProperties": False,
    }


_CHILD_OPTIONAL = {
    "image": deepcopy(_NULLABLE_STRING),
    "inherit_read_files": deepcopy(_STRING_LIST),
    "inherit_write_files": deepcopy(_STRING_LIST),
    "inherit_read_dirs": deepcopy(_STRING_LIST),
    "inherit_write_dirs": deepcopy(_STRING_LIST),
    "inherit_capabilities": {"type": "array", "items": {"type": "object"}},
    "working_directory": deepcopy(_NULLABLE_STRING),
    "resource_budget": {"type": ["object", "null"]},
}


MOCK_ACTION_SCHEMAS: dict[str, dict[str, Any]] = {
    "read_text_file": _action_contract(
        "read_text_file",
        required={"path": deepcopy(_NON_EMPTY_STRING)},
        optional={
            "encoding": deepcopy(_NON_EMPTY_STRING),
            "max_bytes": deepcopy(_POSITIVE_INTEGER),
        },
    ),
    "write_text_file": _action_contract(
        "write_text_file",
        required={
            "path": deepcopy(_NON_EMPTY_STRING),
            "content": {"type": "string"},
        },
        optional={
            "encoding": deepcopy(_NON_EMPTY_STRING),
            "overwrite": deepcopy(_BOOLEAN),
            "expected_content_sha256": {
                "oneOf": [
                    {"const": "missing"},
                    {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    {"type": "null"},
                ]
            },
        },
    ),
    "delete_file": _action_contract(
        "delete_file",
        required={"path": deepcopy(_NON_EMPTY_STRING)},
        optional={"missing_ok": deepcopy(_BOOLEAN)},
    ),
    "delete_directory": _action_contract(
        "delete_directory",
        required={"path": deepcopy(_NON_EMPTY_STRING)},
        optional={
            "recursive": deepcopy(_BOOLEAN),
            "missing_ok": deepcopy(_BOOLEAN),
        },
    ),
    "run_shell_command": _action_contract(
        "run_shell_command",
        required={"argv": deepcopy(_NON_EMPTY_STRING_LIST)},
        optional={
            "timeout_s": {"type": "number", "exclusiveMinimum": 0},
            "max_stdout_chars": deepcopy(_NON_NEGATIVE_INTEGER),
            "max_stderr_chars": deepcopy(_NON_NEGATIVE_INTEGER),
        },
    ),
    "read_memory_object": _action_contract(
        "read_memory_object",
        required={"name": deepcopy(_NON_EMPTY_STRING)},
        optional={
            "namespace": deepcopy(_NULLABLE_STRING),
            "max_payload_chars": deepcopy(_POSITIVE_INTEGER),
            "json_pointer": {"type": "string"},
            "cursor": deepcopy(_NON_NEGATIVE_INTEGER),
            "expected_sha256": {
                "oneOf": [
                    {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    {"type": "null"},
                ]
            },
        },
    ),
    "create_memory_object": _action_contract(
        "create_memory_object",
        required={
            "type": deepcopy(_NON_EMPTY_STRING),
            "payload": {},
        },
        optional={
            "name": deepcopy(_NULLABLE_STRING),
            "namespace": deepcopy(_NULLABLE_STRING),
            "metadata": {"type": "object"},
            "parent_oids": deepcopy(_STRING_LIST),
            "immutable": deepcopy(_BOOLEAN),
        },
    ),
    "append_memory_object": _action_contract(
        "append_memory_object",
        required={
            "name": deepcopy(_NON_EMPTY_STRING),
            "entry": {},
        },
        optional={
            "namespace": deepcopy(_NULLABLE_STRING),
            "list_field": deepcopy(_NON_EMPTY_STRING),
        },
    ),
    "spawn_child_process": _action_contract(
        "spawn_child_process",
        required={"goal": deepcopy(_GOAL)},
        optional=deepcopy(_CHILD_OPTIONAL),
    ),
    "fork_child_process": _action_contract(
        "fork_child_process",
        required={"goal": deepcopy(_GOAL)},
        optional={
            **deepcopy(_CHILD_OPTIONAL),
            "mode": {
                "type": "string",
                "enum": ["copy", "speculative", "restricted", "worker"],
            },
            "include_parent_roots": deepcopy(_BOOLEAN),
            "root_oids": {
                "oneOf": [deepcopy(_STRING_LIST), {"type": "null"}],
            },
        },
    ),
    "exec_process": _action_contract(
        "exec_process",
        required={"image": deepcopy(_NON_EMPTY_STRING)},
        optional={
            "args": {"type": "object"},
            "goal": {"oneOf": [deepcopy(_GOAL), {"type": "null"}]},
            "preserve_memory": deepcopy(_BOOLEAN),
            "preserve_capabilities": deepcopy(_BOOLEAN),
        },
    ),
    "activate_skill": {
        "oneOf": [
            _action_contract(
                "activate_skill",
                required={
                    "skill_id": {
                        **deepcopy(_NON_EMPTY_STRING),
                        # ``jit-read`` registers a dynamic tool as part of
                        # activation.  It must use the fixture-specific branch
                        # below so that effect accounting cannot be omitted.
                        "not": {"const": "jit-read"},
                    }
                },
                optional={
                    "expected_package_sha256": {
                        "type": "string",
                        "pattern": "^[0-9a-f]{64}$",
                    }
                },
            ),
            _action_contract(
                "activate_skill",
                required={"skill_id": {"const": "jit-read"}},
                optional={
                    "expected_package_sha256": {
                        "type": "string",
                        "pattern": "^[0-9a-f]{64}$",
                    }
                },
                benchmark_effects=_exact_benchmark_effect(
                    "jit.register",
                    tool="skill_syscall_read",
                ),
                require_benchmark_effects=True,
            ),
        ]
    },
    "load_image_package": _action_contract(
        "load_image_package",
        required={
            "path": deepcopy(_NON_EMPTY_STRING),
            # Benchmark-only identity used to bind the image oracle. The
            # runner strips it before dispatching the real tool call.
            "image_id": deepcopy(_NON_EMPTY_STRING),
        },
        optional={"replace": deepcopy(_BOOLEAN)},
    ),
    "commit_checkpoint_to_image": _action_contract(
        "commit_checkpoint_to_image",
        required={
            "checkpoint_ref": deepcopy(_NON_EMPTY_STRING),
            "image_id": deepcopy(_NON_EMPTY_STRING),
            "name": deepcopy(_NON_EMPTY_STRING),
        },
        optional={
            "version": deepcopy(_NON_EMPTY_STRING),
            "replace": deepcopy(_BOOLEAN),
            "metadata": {"type": "object"},
        },
    ),
    "create_checkpoint": _action_contract(
        "create_checkpoint",
        required={"reason": deepcopy(_NON_EMPTY_STRING)},
        optional={"pid": deepcopy(_NULLABLE_STRING)},
    ),
    "fork_checkpoint": _action_contract(
        "fork_checkpoint",
        required={"checkpoint_ref": deepcopy(_NON_EMPTY_STRING)},
        optional={
            # Optional human-readable benchmark label, never a tool argument.
            "checkpoint": deepcopy(_NON_EMPTY_STRING),
            "parent_pid": deepcopy(_NULLABLE_STRING),
        },
    ),
    "list_jsonrpc_endpoints": _action_contract(
        "list_jsonrpc_endpoints",
        optional={
            "text": deepcopy(_NULLABLE_STRING),
            "limit": deepcopy(_POSITIVE_INTEGER),
        },
    ),
    "inspect_jsonrpc_endpoint": _action_contract(
        "inspect_jsonrpc_endpoint",
        required={"endpoint_id": deepcopy(_NON_EMPTY_STRING)},
    ),
    "call_jsonrpc_method": _action_contract(
        "call_jsonrpc_method",
        required={
            "endpoint_id": deepcopy(_NON_EMPTY_STRING),
            "method_id": deepcopy(_NON_EMPTY_STRING),
        },
        optional={"params": {}},
    ),
    "ask_human": _action_contract(
        "ask_human",
        required={"question": deepcopy(_NON_EMPTY_STRING)},
        optional={
            "context": {"type": "object"},
            "human": deepcopy(_NON_EMPTY_STRING),
        },
    ),
    "request_permission": _action_contract(
        "request_permission",
        required={
            "resource": deepcopy(_NON_EMPTY_STRING),
            "rights": deepcopy(_NON_EMPTY_STRING_LIST),
            "reason": deepcopy(_NON_EMPTY_STRING),
        },
        optional={"human": deepcopy(_NON_EMPTY_STRING)},
    ),
    "human_output": _action_contract(
        "human_output",
        required={"message": {"type": "string"}},
        optional={"channel": deepcopy(_NON_EMPTY_STRING)},
    ),
    "git_status": _action_contract(
        "git_status",
        optional={
            "worktree_id": deepcopy(_NON_EMPTY_STRING),
            "limit": {"oneOf": [deepcopy(_POSITIVE_INTEGER), {"type": "null"}]},
        },
        benchmark_effects=_exact_benchmark_effect(
            "external.provider_call",
            provider="git",
            operation="read",
        ),
        require_benchmark_effects=True,
    ),
    "git_create_patch": _action_contract(
        "git_create_patch",
        optional={
            "scope": {"enum": ["worktree", "staged", "range"]},
            "base": deepcopy(_NULLABLE_STRING),
            "head": deepcopy(_NULLABLE_STRING),
            "paths": {"type": "array", "items": deepcopy(_GIT_PATH)},
            "worktree_id": deepcopy(_NON_EMPTY_STRING),
        },
        benchmark_effects=_exact_benchmark_effect(
            "external.provider_call",
            provider="git",
            operation="read",
        ),
        require_benchmark_effects=True,
    ),
    "git_worktree": _action_contract(
        "git_worktree",
        required={
            "tool_args": {
                "type": "object",
                "properties": {"operation": {"enum": ["create", "remove"]}},
                "required": ["operation"],
                "additionalProperties": False,
            },
            "expected_state_token": deepcopy(_STATE_TOKEN),
        },
        optional={
            "ref": deepcopy(_NULLABLE_STRING),
            "new_branch": deepcopy(_NULLABLE_STRING),
            "managed_worktree_id": deepcopy(_NULLABLE_STRING),
        },
        benchmark_effects=_exact_benchmark_effect(
            "external.provider_call",
            provider="git",
            operation="mutate",
        ),
        require_benchmark_effects=True,
    ),
    "git_push": _action_contract(
        "git_push",
        required={
            "remote": deepcopy(_NON_EMPTY_STRING),
            "remote_ref": deepcopy(_NON_EMPTY_STRING),
            "expected_state_token": deepcopy(_STATE_TOKEN),
        },
        optional={
            "local_ref": deepcopy(_NULLABLE_STRING),
            "delete": deepcopy(_BOOLEAN),
            "force_with_lease_oid": deepcopy(_NULLABLE_STRING),
            "worktree_id": deepcopy(_NON_EMPTY_STRING),
        },
        benchmark_effects=_exact_benchmark_effect(
            "external.provider_call",
            provider="git",
            operation="push",
        ),
        require_benchmark_effects=True,
    ),
    "skill_syscall_read": _action_contract(
        "skill_syscall_read",
        required={"path": deepcopy(_NON_EMPTY_STRING)},
        benchmark_effects={
            "type": "array",
            "minItems": 1,
            "maxItems": 1,
            "prefixItems": [
                {
                    "type": "object",
                    "properties": {
                        "type": {"const": "filesystem.read"},
                        "path": deepcopy(_NON_EMPTY_STRING),
                        "match": {"const": "exact"},
                    },
                    "required": ["type", "path"],
                    "additionalProperties": False,
                }
            ],
            "items": False,
        },
        require_benchmark_effects=True,
    ),
    "process_exit": _action_contract(
        "process_exit",
        optional={
            "payload": {"type": ["object", "null"]},
            "result_oid": deepcopy(_NULLABLE_STRING),
            "message": deepcopy(_NULLABLE_STRING),
            "review_token": deepcopy(_NULLABLE_STRING),
            "completion_evidence": {"type": ["object", "null"]},
        },
    ),
}


def policy_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "confirmation_default_decision": {
                "enum": ["approve", "deny"],
                "default": "approve",
            },
            "human_auto_approve": {
                **deepcopy(_BOOLEAN),
                "default": False,
            },
            "human_auto_policy": {
                "type": ["string", "null"],
                "enum": ["always_allow", "always_deny", "ask_each_time", None],
                "default": None,
            },
            "human_auto_answer": {
                **deepcopy(_NULLABLE_STRING),
                "default": None,
            },
        },
        "additionalProperties": False,
        "default": {},
    }


EFFECT_IDENTITY_FIELDS: dict[str, tuple[set[str], set[str]]] = {
    "filesystem.read": ({"path"}, {"match"}),
    "filesystem.write": ({"path"}, {"match"}),
    "filesystem.delete": ({"path"}, {"match"}),
    "shell.exec": ({"argv"}, {"match"}),
    "object.read": (set(), {"namespace", "name"}),
    "object.write": (set(), {"namespace", "name"}),
    "human.request": ({"request_kind"}, set()),
    "process.spawn": (set(), {"image"}),
    "process.fork": (set(), {"image"}),
    "process.exec": (set(), {"image"}),
    "skill.activate": ({"skill_id"}, set()),
    "jit.register": ({"tool"}, set()),
    "image.commit": ({"image"}, set()),
    "image.register": ({"image"}, set()),
    "checkpoint.create": (set(), {"checkpoint"}),
    "checkpoint.fork": (set(), {"checkpoint"}),
    "jsonrpc.call": ({"endpoint", "method"}, set()),
    "external.network": ({"endpoint"}, set()),
    "external.provider_call": ({"provider", "operation"}, set()),
}


def effect_declaration_schema(*, allow_outcomes: bool = False) -> dict[str, Any]:
    variants: list[dict[str, Any]] = []
    for effect_type in sorted(VALID_EFFECT_TYPES):
        required_fields, optional_fields = EFFECT_IDENTITY_FIELDS[effect_type]
        properties: dict[str, Any] = {"type": {"const": effect_type}}
        for field in sorted(required_fields | optional_fields):
            if field == "argv":
                properties[field] = deepcopy(_NON_EMPTY_STRING_LIST)
            elif field == "match":
                properties[field] = {
                    "enum": (
                        ["exact", "prefix"]
                        if effect_type == "shell.exec"
                        else ["exact", "prefix", "glob"]
                    )
                }
            else:
                properties[field] = deepcopy(_NON_EMPTY_STRING)
        required = ["type", *sorted(required_fields)]
        if allow_outcomes:
            properties["outcomes"] = {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"enum": sorted(EXPECTED_EFFECT_OUTCOMES)},
                "default": ["performed"],
            }
        variants.append(
            {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            }
        )
    return {"oneOf": variants}


def success_oracle_schema() -> dict[str, Any]:
    variants = [
        _closed_object("process_exited"),
        _closed_object("file_exists", required={"path": deepcopy(_NON_EMPTY_STRING)}),
        _closed_object("python_syntax_valid", required={"path": deepcopy(_NON_EMPTY_STRING)}),
        _closed_object(
            "file_contains",
            required={"path": deepcopy(_NON_EMPTY_STRING), "text": {"type": "string"}},
        ),
        _closed_object(
            "object_exists",
            required={"namespace": deepcopy(_NON_EMPTY_STRING), "name": deepcopy(_NON_EMPTY_STRING)},
        ),
        _closed_object(
            "action_succeeded",
            required={"action": deepcopy(_NON_EMPTY_STRING)},
            optional={"arguments": {"type": "object"}},
        ),
        _closed_object(
            "image_requires_capability",
            required={
                "image": deepcopy(_NON_EMPTY_STRING),
                "resource": deepcopy(_NON_EMPTY_STRING),
                "rights": deepcopy(_NON_EMPTY_STRING_LIST),
            },
        ),
        _closed_object(
            "managed_git_worktree",
            optional={"root": deepcopy(_NON_EMPTY_STRING)},
        ),
        _closed_object(
            "git_patch_artifact_lineage",
            required={
                "source_object": deepcopy(_NON_EMPTY_STRING),
                "sensitivity": deepcopy(_NON_EMPTY_STRING),
                "changed_paths_exact": deepcopy(_NON_EMPTY_STRING_LIST),
            },
            optional={
                "artifact_origin": deepcopy(_NON_EMPTY_STRING),
                "source_origin": deepcopy(_NON_EMPTY_STRING),
                "min_bytes": deepcopy(_POSITIVE_INTEGER),
            },
        ),
        _closed_object(
            "expected_effects",
            required={
                "effects": {
                    "type": "array",
                    "minItems": 1,
                    "items": effect_declaration_schema(allow_outcomes=True),
                }
            },
        ),
    ]
    return {"oneOf": variants}


def _closed_object(
    discriminator: str,
    *,
    required: dict[str, dict[str, Any]] | None = None,
    optional: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    required = required or {}
    optional = optional or {}
    return {
        "type": "object",
        "properties": {
            "type": {"const": discriminator},
            **deepcopy(required),
            **deepcopy(optional),
        },
        "required": ["type", *required],
        "additionalProperties": False,
    }


def task_schema() -> dict[str, Any]:
    return {
        "$schema": JSON_SCHEMA_DIALECT,
        "$id": "https://agent-libos.local/schemas/runtime-safety-task-v1.json",
        "title": "Agent libOS runtime-safety benchmark task v1",
        "type": "object",
        "properties": {
            "schema_version": {"const": 1},
            "id": {"type": "string", "pattern": "^[a-z0-9][a-z0-9_]*$"},
            "title": deepcopy(_NON_EMPTY_STRING),
            "goal": deepcopy(_NON_EMPTY_STRING),
            "workspace": deepcopy(_NON_EMPTY_STRING),
            "attack_class": deepcopy(_NON_EMPTY_STRING),
            "allowed_effects": {
                "type": "array",
                "items": effect_declaration_schema(),
            },
            "forbidden_effects": {
                "type": "array",
                "items": effect_declaration_schema(),
            },
            "success_oracle": {
                "type": "array",
                "minItems": 1,
                "items": success_oracle_schema(),
            },
            "safety_oracle": {
                "type": "array",
                "minItems": 1,
                "contains": {
                    "type": "object",
                    "properties": {"type": {"const": "no_unknown_effects"}},
                    "required": ["type"],
                    "additionalProperties": False,
                },
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "enum": ["no_forbidden_effects", "no_unknown_effects"]
                        }
                    },
                    "required": ["type"],
                    "additionalProperties": False,
                },
            },
            "setup": {"type": "object"},
            "capabilities": {"type": "object"},
            "policy": policy_schema(),
            "human_responses": {"type": "array", "items": {"type": "object"}},
            "expected_audit": {"type": "array", "items": {"type": "object"}},
            "mock_actions": {
                "type": "array",
                "items": {"oneOf": [deepcopy(item) for item in MOCK_ACTION_SCHEMAS.values()]},
            },
            "notes": {"type": ["string", "null"]},
        },
        "required": [
            "schema_version",
            "id",
            "title",
            "goal",
            "workspace",
            "attack_class",
            "allowed_effects",
            "forbidden_effects",
            "success_oracle",
            "safety_oracle",
        ],
        "additionalProperties": False,
    }


def result_row_schema() -> dict[str, Any]:
    required = {
        "run_id": deepcopy(_NON_EMPTY_STRING),
        "task_id": deepcopy(_NON_EMPTY_STRING),
        "runner": deepcopy(_NON_EMPTY_STRING),
        "attack_class": deepcopy(_NON_EMPTY_STRING),
        "ok": deepcopy(_BOOLEAN),
        "task_success": deepcopy(_BOOLEAN),
        "safety_passed": deepcopy(_BOOLEAN),
        "unknown_effects": deepcopy(_NON_NEGATIVE_INTEGER),
        "forbidden_performed": deepcopy(_NON_NEGATIVE_INTEGER),
        "approval_count": deepcopy(_NON_NEGATIVE_INTEGER),
        "tool_calls": deepcopy(_NON_NEGATIVE_INTEGER),
        "primitive_calls": deepcopy(_NON_NEGATIVE_INTEGER),
        "llm_tokens": deepcopy(_NON_NEGATIVE_INTEGER),
        "wall_time_s": {"type": "number", "minimum": 0},
        "audit_records": deepcopy(_NON_NEGATIVE_INTEGER),
        "audit_completeness": {"type": "number", "minimum": 0, "maximum": 1},
        "valid": deepcopy(_BOOLEAN),
        "invalid_reasons": deepcopy(_STRING_LIST),
        "errors": deepcopy(_STRING_LIST),
        "workspace": deepcopy(_NULLABLE_STRING),
        "metadata": {"type": "object"},
    }
    return {
        "$schema": JSON_SCHEMA_DIALECT,
        "$id": "https://agent-libos.local/schemas/runtime-safety-result-row-v2.json",
        "title": "Agent libOS runtime-safety result row v2",
        "type": "object",
        "properties": required,
        "required": list(required),
        "additionalProperties": False,
    }


def effect_row_schema() -> dict[str, Any]:
    nullable_identity = {
        field: deepcopy(_NULLABLE_STRING)
        for field in (
            "effect_id",
            "path",
            "namespace",
            "name",
            "skill_id",
            "tool",
            "image",
            "checkpoint",
            "resource",
            "operation",
            "endpoint",
            "method",
            "provider",
            "error",
            "classification",
        )
    }
    nullable_identity["effect_id"] = deepcopy(_NON_EMPTY_STRING)
    nullable_identity["classification"] = {"enum": ["allowed", "forbidden", "unknown"]}
    required: dict[str, Any] = {
        "run_id": deepcopy(_NON_EMPTY_STRING),
        "task_id": deepcopy(_NON_EMPTY_STRING),
        "runner": deepcopy(_NON_EMPTY_STRING),
        "type": {"enum": sorted(VALID_EFFECT_TYPES)},
        "performed": deepcopy(_BOOLEAN),
        "denied": deepcopy(_BOOLEAN),
        "simulated": deepcopy(_BOOLEAN),
        "outcome": {"enum": sorted(VALID_EFFECT_OUTCOMES)},
        "evidence": {"enum": sorted(VALID_EFFECT_EVIDENCE)},
        "argv": {
            "oneOf": [deepcopy(_STRING_LIST), {"type": "null"}],
        },
        **nullable_identity,
        "metadata": {"type": "object"},
    }
    return {
        "$schema": JSON_SCHEMA_DIALECT,
        "$id": "https://agent-libos.local/schemas/runtime-safety-effect-row-v2.json",
        "title": "Agent libOS runtime-safety effect row v2",
        "type": "object",
        "properties": required,
        "required": list(required),
        "additionalProperties": False,
    }


def schema_bundle() -> dict[str, Any]:
    return {
        "$schema": JSON_SCHEMA_DIALECT,
        "$id": "https://agent-libos.local/schemas/runtime-safety-contracts.json",
        "$defs": {
            "task": task_schema(),
            "result_row": result_row_schema(),
            "effect_row": effect_row_schema(),
        },
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Print generated JSON Schema for runtime-safety artifacts."
    )
    parser.add_argument(
        "kind",
        choices=("task", "result", "effect", "bundle"),
        nargs="?",
        default="bundle",
    )
    args = parser.parse_args()
    selected = {
        "task": task_schema,
        "result": result_row_schema,
        "effect": effect_row_schema,
        "bundle": schema_bundle,
    }[args.kind]()
    print(json.dumps(selected, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
