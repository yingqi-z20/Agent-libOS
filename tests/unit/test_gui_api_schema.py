from __future__ import annotations

import ast
import json
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "docs" / "gui_api_schema.json"
SERVER_PATH = ROOT / "agent_libos" / "api" / "gui" / "server.py"


def _schema() -> dict[str, object]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _validator_for(definition: str) -> Draft202012Validator:
    schema = _schema()
    return Draft202012Validator(
        {
            "$schema": schema["$schema"],
            "$defs": schema["$defs"],
            "$ref": f"#/$defs/{definition}",
        }
    )


_ROUTE_BASES = {
    "_dispatch_workflows": ("api", "workflows"),
    "_dispatch_process": ("api", "processes", "{pid}"),
    "_dispatch_checkpoints": ("api", "checkpoints"),
    "_dispatch_skills": ("api", "skills"),
    "_dispatch_capabilities": ("api", "capabilities"),
    "_dispatch_images": ("api", "images"),
    "_dispatch_jsonrpc": ("api", "jsonrpc"),
    "_dispatch_mcp": ("api", "mcp"),
}
_ROUTE_PLACEHOLDERS = {
    "_dispatch_checkpoints": {0: "{checkpoint_id}"},
    "_dispatch_skills": {0: "{skill_id}"},
    "_dispatch_capabilities": {0: "{capability_id}"},
    "_dispatch_jsonrpc": {0: "{endpoint_id}"},
    "_dispatch_mcp": {0: "{server_id}"},
}


def _confirmed_contracts_from_server() -> dict[str, str]:
    tree = ast.parse(SERVER_PATH.read_text(encoding="utf-8"))
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    contracts: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "_require_confirmed" or not node.args:
            continue
        actions = _confirmed_actions(node.args[0])
        function = _ancestor(node, parents, ast.FunctionDef)
        route_guard = _route_guard(node, parents)
        assert isinstance(function, ast.FunctionDef)
        assert route_guard is not None
        for action in actions:
            relative = _relative_route(route_guard.test, function.name, action)
            route = "/".join((*_ROUTE_BASES[function.name], *relative))
            contracts[action] = f"POST /{route}"
    return contracts


def _confirmed_actions(action: ast.expr) -> set[str]:
    if isinstance(action, ast.Constant) and isinstance(action.value, str):
        return {action.value}
    if isinstance(action, ast.JoinedStr):
        literal = "".join(
            value.value
            for value in action.values
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
        )
        if literal == "skill.":
            return {"skill.activate", "skill.unload"}
    raise AssertionError("unsupported dynamic confirmation action")


def _ancestor(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    kind: type[ast.AST],
) -> ast.AST | None:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, kind):
            return current
        current = parents.get(current)
    return None


def _route_guard(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.If | None:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, ast.If) and any(
            isinstance(item, ast.Name) and item.id == "route"
            for item in ast.walk(current.test)
        ):
            return current
        current = parents.get(current)
    return None


def _relative_route(test: ast.expr, function: str, action: str) -> tuple[str, ...]:
    for item in ast.walk(test):
        if not isinstance(item, ast.Compare) or len(item.ops) != 1:
            continue
        if not isinstance(item.left, ast.Name) or item.left.id != "route":
            continue
        value = item.comparators[0]
        if isinstance(item.ops[0], ast.Eq) and isinstance(value, (ast.List, ast.Tuple)):
            return tuple(
                element.value
                for element in value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            )

    indexed: dict[int, str] = dict(_ROUTE_PLACEHOLDERS.get(function, {}))
    route_length = 0
    for item in ast.walk(test):
        if not isinstance(item, ast.Compare) or len(item.ops) != 1:
            continue
        if (
            isinstance(item.left, ast.Call)
            and isinstance(item.left.func, ast.Name)
            and item.left.func.id == "len"
            and len(item.left.args) == 1
            and isinstance(item.left.args[0], ast.Name)
            and item.left.args[0].id == "route"
            and isinstance(item.comparators[0], ast.Constant)
            and isinstance(item.comparators[0].value, int)
        ):
            route_length = item.comparators[0].value
            continue
        if not (
            isinstance(item.left, ast.Subscript)
            and isinstance(item.left.value, ast.Name)
            and item.left.value.id == "route"
            and isinstance(item.left.slice, ast.Constant)
            and isinstance(item.left.slice.value, int)
        ):
            continue
        index = item.left.slice.value
        comparator = item.comparators[0]
        if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
            indexed[index] = comparator.value
        elif isinstance(item.ops[0], ast.In) and isinstance(comparator, (ast.Set, ast.Tuple)):
            choices = {
                element.value
                for element in comparator.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            }
            selected = action.rsplit(".", 1)[-1]
            assert selected in choices
            indexed[index] = selected
    assert route_length > 0
    assert set(indexed) == set(range(route_length))
    return tuple(indexed[index] for index in range(route_length))


def test_gui_api_schema_is_valid_draft_2020_12() -> None:
    schema = _schema()
    Draft202012Validator.check_schema(schema)
    assert schema["x-agent-libos-schema-version"] == 1
    assert list(Draft202012Validator(schema).iter_errors({"confirmed": True}))
    usage = schema["x-agent-libos-definition-usage"]
    assert usage["local_ref_example"].endswith("#/$defs/processExecPayload")


def test_gui_api_schema_tracks_every_explicit_confirmation_operation() -> None:
    schema = _schema()
    scope = schema["x-agent-libos-contract-scope"]
    operations = scope["confirmed_operations"]
    documented = set(operations)
    server_contracts = _confirmed_contracts_from_server()

    assert documented == set(server_contracts)
    for operation, contract in operations.items():
        assert contract["route"] == server_contracts[operation], operation
        assert contract["request_def"] in schema["$defs"], operation


def test_gui_api_schema_validates_snapshot_and_error_envelopes() -> None:
    snapshot = {
        "db": "local",
        "scheduler": {"auto_run": True, "running": False, "paused": False},
        "processes": [{"pid": "pid_1", "status": "waiting"}],
        "human_requests": [],
        "events": [],
        "audit": [],
        "llm_calls": [],
        "object_tasks": [],
        "tools": [],
        "images": [],
        "skills": [],
        "jsonrpc_endpoints": [],
        "mcp_servers": [],
        "modules": [],
        "llm_profiles": [],
    }
    snapshot_validator = _validator_for("snapshotResponse")
    snapshot_validator.validate(snapshot)
    invalid_snapshot = {**snapshot, "events": [42]}
    assert list(snapshot_validator.iter_errors(invalid_snapshot))
    _validator_for("errorEnvelope").validate(
        {
            "ok": False,
            "error": {
                "message": "process.exec requires explicit confirmation",
                "confirmation_required": True,
                "action": "process.exec",
                "preview": {"pid": "pid_1"},
            },
        }
    )


def test_gui_api_schema_requires_confirmation_and_workspace_relative_skill_path() -> None:
    workflow = _validator_for("workflowRunPayload")
    workflow.validate({"tool": "get_working_directory", "args": {}})
    workflow.validate(
        {
            "tool": "get_working_directory",
            "args": {},
            "confirmed": False,
        }
    )
    for field in ("image", "working_directory"):
        assert list(
            workflow.iter_errors(
                {"tool": "get_working_directory", "args": {}, field: 7}
            )
        )

    process_exec = _validator_for("processExecPayload")
    assert list(process_exec.iter_errors({"image": "review:v0"}))
    process_exec.validate({"confirmed": True, "image": "review:v0"})
    process_exec.validate(
        {
            "confirmed": True,
            "image": "review:v0",
            "goal": {"task": "review", "target": "README.md"},
        }
    )
    assert list(
        process_exec.iter_errors(
            {"confirmed": True, "image": "review:v0", "args": []}
        )
    )
    assert list(
        process_exec.iter_errors(
            {"confirmed": True, "actor": "pid_1", "image": "review:v0"}
        )
    )

    process_exit = _validator_for("processExitPayload")
    process_exit.validate({"confirmed": True, "message": None})
    process_exit.validate({"confirmed": True, "message": "done"})
    assert list(
        process_exit.iter_errors({"confirmed": True, "message": {"text": "done"}})
    )

    skill_register = _validator_for("skillRegisterPayload")
    skill_register.validate(
        {"confirmed": True, "actor": "pid_1", "path": "skills/reviewer"}
    )
    assert list(
        skill_register.iter_errors(
            {"confirmed": True, "path": "skills/reviewer"}
        )
    )
    assert list(
        skill_register.iter_errors(
            {"confirmed": True, "actor": "pid_1", "path": "/tmp/reviewer"}
        )
    )
    for unsafe_path in (r"C:\tmp\reviewer", r"\\server\reviewer", r"skills\..\reviewer"):
        assert list(
            skill_register.iter_errors(
                {"confirmed": True, "actor": "pid_1", "path": unsafe_path}
            )
        )

    skill_process_mutation = _validator_for("skillProcessMutationPayload")
    skill_process_mutation.validate({"confirmed": True, "pid": "pid_1"})
    skill_process_mutation.validate(
        {"confirmed": True, "pid": "pid_1", "actor": "pid_1"}
    )
    skill_activate = _validator_for("skillActivatePayload")
    package_sha256 = "b" * 64
    skill_activate.validate(
        {
            "confirmed": True,
            "pid": "pid_1",
            "actor": "pid_1",
            "expected_package_sha256": package_sha256,
        }
    )
    assert list(
        skill_activate.iter_errors(
            {"confirmed": True, "pid": "pid_1", "actor": "pid_1"}
        )
    )
    for invalid_hash in ("", "b" * 63, "B" * 64, "not-a-sha256"):
        assert list(
            skill_activate.iter_errors(
                {
                    "confirmed": True,
                    "pid": "pid_1",
                    "expected_package_sha256": invalid_hash,
                }
            )
        )

    _validator_for("capabilityDelegatePayload").validate(
        {
            "confirmed": True,
            "actor": "pid_parent",
            "parent": "pid_parent",
            "child": "pid_child",
            "resource": "filesystem:workspace:docs",
        }
    )

    _validator_for("imageCommitPayload").validate(
        {
            "confirmed": True,
            "checkpoint_id": "cp_1",
            "image_id": "reviewer:v0",
            "name": "reviewer",
        }
    )
    assert list(
        _validator_for("imageRegisterPayload").iter_errors(
            {"confirmed": True, "files": {"IMAGE.yaml": 7}}
        )
    )
    image_register = _validator_for("imageRegisterPayload")
    image_register.validate(
        {
            "confirmed": True,
            "source": "selected-package",
            "files": {"IMAGE.yaml": "schema_version: 1"},
        }
    )
    assert list(
        image_register.iter_errors(
            {
                "confirmed": True,
                "source": "selected-package",
                "files": {"IMAGE.yaml": "schema_version: 1"},
                "path": "/tmp/package",
            }
        )
    )
    assert list(
        _validator_for("registryManifestPayload").iter_errors(
            {"confirmed": True, "manifest_text": "schema_version: 1", "path": "server.yaml"}
        )
    )

    mcp_call = _validator_for("mcpCallPayload")
    mcp_call.validate(
        {"confirmed": True, "pid": "pid_1", "tool_id": "echo", "arguments": None}
    )
    assert list(
        mcp_call.iter_errors(
            {
                "confirmed": True,
                "actor": "pid_1",
                "pid": "pid_1",
                "tool_id": "echo",
            }
        )
    )
    assert list(
        mcp_call.iter_errors(
            {"confirmed": True, "pid": "pid_1", "tool_id": "echo", "arguments": []}
        )
    )
