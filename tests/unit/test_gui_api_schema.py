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


def _llm_call_summary() -> dict[str, object]:
    return {
        "schema_version": 1,
        "call_id": "llmcall_1",
        "pid": "pid_1",
        "image_id": "base-agent:v0",
        "purpose": "action_selection",
        "status": "ok",
        "api": "responses",
        "model": "provider-model",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
        },
        "error": None,
        "created_at": "2030-01-01T00:00:00Z",
        "completed_at": "2030-01-01T00:00:01Z",
        "request_id": "req_1",
        "response_id": "resp_1",
        "attempt_count": 2,
        "coverage": "complete",
        "selected_attempt": 2,
        "reasoning_availability": "returned",
        "payload_retention_tier": "full",
    }


_ROUTE_BASES = {
    "_dispatch_workflows": ("api", "workflows"),
    "_dispatch_process": ("api", "processes", "{pid}"),
    "_dispatch_checkpoints": ("api", "checkpoints"),
    "_dispatch_skills": ("api", "skills"),
    "_dispatch_capabilities": ("api", "capabilities"),
    "_dispatch_images": ("api", "images"),
    "_dispatch_jsonrpc": ("api", "jsonrpc"),
    "_dispatch_mcp": ("api", "mcp"),
    "_dispatch_task_runs": ("api", "task-runs"),
}
_ROUTE_PLACEHOLDERS = {
    "_dispatch_checkpoints": {0: "{checkpoint_id}"},
    "_dispatch_skills": {0: "{skill_id}"},
    "_dispatch_capabilities": {0: "{capability_id}"},
    "_dispatch_jsonrpc": {0: "{endpoint_id}"},
    "_dispatch_mcp": {0: "{server_id}"},
    "_dispatch_task_runs": {0: "{run_id}"},
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
        function, route_guard = _confirmed_route_context(node, tree, parents)
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


def _confirmed_route_context(
    confirmation: ast.Call,
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
) -> tuple[ast.FunctionDef, ast.If]:
    """Resolve a confirmation directly or through one routed action helper.

    Extracted route bodies keep confirmation behavior close to the action while
    the dispatcher retains the HTTP contract.  Require exactly one guarded
    dispatcher callsite so an unguarded or ambiguously reused helper still
    fails this contract check instead of silently losing coverage.
    """

    function = _ancestor(confirmation, parents, ast.FunctionDef)
    assert isinstance(function, ast.FunctionDef)
    direct_guard = _route_guard(confirmation, parents)
    if direct_guard is not None:
        assert function.name in _ROUTE_BASES
        return function, direct_guard

    routed_callsites: list[tuple[ast.FunctionDef, ast.If]] = []
    for candidate in ast.walk(tree):
        if not isinstance(candidate, ast.Call):
            continue
        if not isinstance(candidate.func, ast.Attribute):
            continue
        if candidate.func.attr != function.name:
            continue
        caller = _ancestor(candidate, parents, ast.FunctionDef)
        route_guard = _route_guard(candidate, parents)
        if (
            isinstance(caller, ast.FunctionDef)
            and caller.name in _ROUTE_BASES
            and route_guard is not None
        ):
            routed_callsites.append((caller, route_guard))
    assert len(routed_callsites) == 1, (
        f"confirmation helper {function.name} must have exactly one guarded "
        f"dispatcher callsite, found {len(routed_callsites)}"
    )
    return routed_callsites[0]


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
    assert schema["x-agent-libos-schema-version"] == 2
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
        "schema_version": 3,
        "db": "local",
        "scheduler": {"auto_run": True, "running": False, "paused": False},
        "processes": [{"pid": "pid_1", "status": "waiting"}],
        "human_requests": [],
        "events": [],
        "audit": [],
        "llm_calls": [_llm_call_summary()],
        "object_tasks": [],
        "task_runs": [
            {
                "schema_version": 1,
                "run_id": "run_1",
                "revision": 2,
                "status": "paused",
                "display_title": "Durable task",
                "root_pid": "pid_1",
                "active_pid": "pid_1",
                "allowed_actions": ["resume", "cancel"],
                "blockers": [],
                "retention": "purge_on_terminal",
                "payloads_purged": False,
            }
        ],
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
    private_summary = {
        **snapshot["task_runs"][0],
        "goal": "must not cross summary boundary",
    }
    assert list(_validator_for("taskRunSummary").iter_errors(private_summary))
    missing_purge_state = dict(snapshot["task_runs"][0])
    missing_purge_state.pop("payloads_purged")
    assert list(_validator_for("taskRunSummary").iter_errors(missing_purge_state))
    _validator_for("taskRunDetailResponse").validate(
        {
            "summary": snapshot["task_runs"][0],
            "requirements": {
                "items": [
                    {
                        "schema_version": 1,
                        "requirement_id": "req_1",
                        "run_id": "run_1",
                        "ordinal": 0,
                        "kind": "initial",
                        "status": "pending",
                        "requirement_sha256": "a" * 64,
                        "label": "Initial requirement",
                        "created_by": "host",
                        "created_at": "2030-01-01T00:00:00Z",
                        "updated_at": "2030-01-01T00:00:00Z",
                        "started_at": None,
                        "completed_at": None,
                        "waived_by": None,
                        "content_available": False,
                        "content_retention": "hash_only",
                        "content_sha256": "a" * 64,
                    }
                ],
                "next_cursor": None,
                "has_more": False,
            },
            "recovery_options": [],
        }
    )
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
    _validator_for("errorEnvelope").validate(
        {
            "ok": False,
            "error": {
                "type": "TaskRunRevisionConflict",
                "code": "task_run_revision_conflict",
                "message": "stale TaskRun revision",
                "command_admitted": False,
                "current_summary": snapshot["task_runs"][0],
            },
        }
    )


def test_gui_api_schema_llm_call_summary_is_strict_and_content_free() -> None:
    validator = _validator_for("llmCallSummary")
    summary = _llm_call_summary()
    validator.validate(summary)

    for content_field, content in {
        "messages": [{"role": "user", "content": "private prompt"}],
        "tools": [{"name": "private_tool"}],
        "request_options": {"authorization": "private"},
        "reasoning": {"summary": "private reasoning"},
        "response_content": "private output",
        "output": "private output",
        "tool_calls": [{"name": "private_tool", "arguments": {"secret": True}}],
        "tool_arguments": {"secret": True},
        "raw_response": {"opaque": "private provider response"},
    }.items():
        assert list(
            validator.iter_errors({**summary, content_field: content})
        ), content_field

    for required_field in summary:
        invalid = dict(summary)
        invalid.pop(required_field)
        assert list(validator.iter_errors(invalid)), required_field

    assert list(
        validator.iter_errors(
            {**summary, "usage": {"prompt_tokens": "private prompt"}}
        )
    )
    assert list(
        validator.iter_errors(
            {**summary, "usage": {"private_provider_field": 1}}
        )
    )


def test_gui_api_schema_bounds_stale_execution_wait_to_diagnostic_hashes() -> None:
    validator = _validator_for("processWaitState")
    receipt = {
        "schema_version": 1,
        "kind": "stale_execution",
        "pid": "pid_1",
        "recovered_by_owner_sha256": "a" * 64,
        "prior_owner_sha256": "b" * 64,
        "prior_lease_sha256": "c" * 64,
        "prior_execution_generation": 4,
        "recovered_execution_generation": 5,
        "recovered_state_generation": 9,
    }
    validator.validate(receipt)
    validator.validate(
        {
            **receipt,
            "prior_owner_sha256": None,
            "prior_lease_sha256": None,
        }
    )

    for field in (
        "execution_owner_id",
        "execution_lease_id",
        "prior_execution_owner_id",
        "prior_execution_lease_id",
        "runtime_epoch",
        "binding_hash",
        "safe_point",
    ):
        assert list(validator.iter_errors({**receipt, field: "raw-or-extra"}))

    for field in (
        "recovered_by_owner_sha256",
        "prior_owner_sha256",
        "prior_lease_sha256",
    ):
        assert list(validator.iter_errors({**receipt, field: "A" * 64}))
        assert list(validator.iter_errors({**receipt, field: "a" * 63}))

    for missing in receipt:
        invalid = dict(receipt)
        invalid.pop(missing)
        assert list(validator.iter_errors(invalid))

    for field in (
        "prior_execution_generation",
        "recovered_execution_generation",
        "recovered_state_generation",
    ):
        assert list(validator.iter_errors({**receipt, field: -1}))

    schema = _schema()
    stale_branch = next(
        branch
        for branch in schema["$defs"]["processWaitState"]["oneOf"]
        if branch.get("properties", {}).get("kind", {}).get("const")
        == "stale_execution"
    )
    assert "not independent TaskRun resume authority" in stale_branch["description"]


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
    task_run_cancel = _validator_for("taskRunCancelPayload")
    task_run_cancel.validate(
        {
            "confirmed": True,
            "expected_revision": 3,
            "command_id": "cancel-1",
            "reason": "operator request",
        }
    )
    assert list(
        task_run_cancel.iter_errors(
            {"expected_revision": 3, "command_id": "cancel-1"}
        )
    )
    _validator_for("taskRunRecoverPayload").validate(
        {
            "confirmed": True,
            "expected_revision": 4,
            "command_id": "recover-1",
            "option_id": "register_receipt",
            "receipt": {"receipt_id": "r-1"},
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
