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
    "_dispatch_mcp_modern_post": ("api", "mcp"),
    "_dispatch_mcp_modern_prompt_post": ("api", "mcp"),
    "_dispatch_mcp_modern_oauth_post": ("api", "mcp"),
    "_dispatch_mcp_modern_continuation_post": ("api", "mcp"),
    "_dispatch_mcp_modern_remote_task_post": ("api", "mcp"),
    "_dispatch_mcp_modern_subscription_post": ("api", "mcp"),
    "_dispatch_mcp_oauth_profile_admin_post": ("api", "mcp"),
    "_dispatch_task_runs": ("api", "task-runs"),
}


def _route_placeholders(function: str, action: str) -> dict[int, str]:
    placeholders = dict(_ROUTE_PLACEHOLDERS.get(function, {}))
    if not (
        function == "_dispatch_mcp_modern_post"
        or function.startswith("_dispatch_mcp_modern_")
        or function == "_dispatch_mcp_oauth_profile_admin_post"
    ):
        return placeholders
    if action in {"mcp.auth.profile.add", "mcp.auth.profile.replace"}:
        return {}
    if action == "mcp.auth.profile.remove":
        return {2: "{profile_id}"}
    if action.startswith("mcp.auth."):
        return {1: "{profile_id}"}
    if action.startswith("mcp.continuation."):
        return {1: "{continuation_id}"}
    if action.startswith("mcp.remote_task."):
        return {1: "{task_ref}"}
    if action == "mcp.subscription.stop":
        return {1: "{subscription_id}"}
    if action in {"mcp.prompt.confirm", "mcp.subscription.start"}:
        return {0: "{server_id}"}
    raise AssertionError(f"missing MCP modern route placeholder for {action}")
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
        if literal == "mcp.continuation.":
            return {"mcp.continuation.respond", "mcp.continuation.cancel"}
        if literal == "mcp.remote_task.":
            return {"mcp.remote_task.update", "mcp.remote_task.cancel"}
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

    indexed = _route_placeholders(function, action)
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
        ):
            continue
        comparator = item.comparators[0]
        if (
            isinstance(item.left.slice, ast.Slice)
            and isinstance(item.ops[0], ast.Eq)
            and isinstance(comparator, (ast.List, ast.Tuple))
        ):
            route_slice = item.left.slice
            assert route_slice.step is None
            assert route_slice.lower is None or (
                isinstance(route_slice.lower, ast.Constant)
                and isinstance(route_slice.lower.value, int)
            )
            lower = 0 if route_slice.lower is None else route_slice.lower.value
            for offset, element in enumerate(comparator.elts):
                assert isinstance(element, ast.Constant) and isinstance(
                    element.value, str
                )
                indexed[lower + offset] = element.value
            continue
        if not (
            isinstance(item.left.slice, ast.Constant)
            and isinstance(item.left.slice.value, int)
        ):
            continue
        index = item.left.slice.value
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


def test_gui_api_schema_models_durable_mcp_reload_without_weakening_cas() -> None:
    schema = _schema()
    operations = schema["x-agent-libos-contract-scope"]["mcp_v3_host_operations"]
    assert operations["continuation.inspect"] == {
        "route": "POST /api/mcp/continuations/{continuation_id}/inspect",
        "request_def": "mcpEmptyPayload",
        "response_def": "mcpInputRequiredProjection",
        "provider_io": False,
    }
    _validator_for("mcpEmptyPayload").validate({})
    assert list(_validator_for("mcpEmptyPayload").iter_errors({"state": "private"}))

    revision_read = _validator_for("mcpRevisionReadPayload")
    for body in ({}, {"expected_revision": None}, {"expected_revision": 9}):
        revision_read.validate(body)
    for body in (
        {"expected_revision": -1},
        {"expected_revision": True},
        {"expected_revision": 1, "confirmed": True},
    ):
        assert list(revision_read.iter_errors(body))
    mutation = _validator_for("mcpRevisionMutationPayload")
    assert list(mutation.iter_errors({"confirmed": True}))
    mutation.validate({"expected_revision": 9, "confirmed": True})

    input_required = _validator_for("mcpInputRequiredProjection")
    respondable = {
        "kind": "input_required",
        "continuation_id": "continuation-local",
        "revision": 3,
        "respondable": True,
        "input_requests": [],
        "human_request_id": "human-local",
        "human_revision": 4,
        "human_preview_sha256": "a" * 64,
    }
    input_required.validate(respondable)
    unsupported = {
        "kind": "input_required",
        "continuation_id": "",
        "revision": 0,
        "respondable": False,
        "input_requests": [{
            "request_id": "sampling-local",
            "kind": "sampling_unsupported",
            "schema": {},
        }],
        "human_request_id": None,
        "human_revision": None,
        "human_preview_sha256": None,
    }
    input_required.validate(unsupported)
    assert list(input_required.iter_errors({
        **respondable,
        "human_request_id": None,
    }))
    assert list(input_required.iter_errors({
        **unsupported,
        "continuation_id": "forged",
    }))
    assert list(input_required.iter_errors({
        **unsupported,
        "input_requests": [{
            "request_id": "elicitation-local",
            "kind": "elicitation",
            "mode": "form",
            "prompt": "not typed unsupported",
            "schema": {"type": "object", "properties": {}},
        }],
    }))


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
    assert list(
        _validator_for("registryManifestPayload").iter_errors(
            {
                "confirmed": True,
                "manifest_text": "schema_version: 1",
                "source": 7,
            }
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
    mcp_unregister = _validator_for("mcpUnregisterPayload")
    mcp_unregister.validate({"confirmed": True})
    mcp_unregister.validate({"confirmed": True, "actor": "pid_1"})
    assert list(mcp_unregister.iter_errors({}))
    assert list(
        mcp_unregister.iter_errors(
            {"confirmed": True, "actor_pid": "pid_1"}
        )
    )


def test_gui_api_schema_covers_strict_mcp_v3_host_surfaces() -> None:
    page = _validator_for("mcpPagePayload")
    page.validate({})
    page.validate({"cursor": "opaque"})
    assert list(page.iter_errors({"cursor": ""}))
    assert list(page.iter_errors({"url": "https://hidden.invalid"}))

    resource = _validator_for("mcpResourceReadPayload")
    resource.validate({"resource_id": "logical-doc", "variables": {"page": "1"}})
    assert list(resource.iter_errors({"resource_id": "logical-doc", "variables": {"page": 1}}))
    assert list(resource.iter_errors({"resource_id": "logical-doc", "headers": {"x": "y"}}))

    preview = _validator_for("mcpPromptGetPayload")
    preview.validate({"prompt_id": "review", "confirmed": False})
    preview.validate({
        "prompt_id": "review",
        "arguments": {"topic": "MCP"},
        "confirmed": True,
        "expected_preview_sha256": "a" * 64,
    })
    assert list(preview.iter_errors({"prompt_id": "review", "confirmed": True}))
    assert list(preview.iter_errors({
        "prompt_id": "review",
        "confirmed": False,
        "expected_preview_sha256": "a" * 64,
    }))
    assert list(preview.iter_errors({"prompt_id": "review", "arguments": None, "actor": "pid_1"}))

    completion = _validator_for("mcpCompletionPayload")
    completion.validate({
        "reference_type": "ref/prompt",
        "reference_id": "review",
        "argument": {"name": "topic", "value": "MCP"},
        "context": {"tenant": "local"},
    })
    assert list(completion.iter_errors({
        "reference_type": "ref/prompt",
        "reference_id": "review",
        "argument": {"name": "topic", "value": "MCP"},
        "context": {"count": 1},
    }))

    oauth = _validator_for("mcpOAuthLoginPayload")
    oauth.validate({"confirmed": True, "scopes": ["resource.read"]})
    assert list(oauth.iter_errors({"confirmed": False}))
    assert list(oauth.iter_errors({"confirmed": True, "client_secret": "private"}))

    continuation = _validator_for("mcpContinuationRespondPayload")
    human_receipt = {
        "human_request_id": "human-local",
        "human_expected_revision": 2,
        "human_preview_sha256": "c" * 64,
    }
    continuation.validate({
        "expected_revision": 1,
        "responses": {"request-local": "yes"},
        **human_receipt,
        "confirmed": True,
    })
    assert list(continuation.iter_errors({"expected_revision": 1, "responses": {}, "confirmed": False}))
    assert list(continuation.iter_errors({
        "expected_revision": 1,
        "responses": {"request-local": "yes"},
        **human_receipt,
        "human_preview_sha256": "C" * 64,
        "confirmed": True,
    }))

    task_update = _validator_for("mcpRemoteTaskUpdatePayload")
    task_update.validate({
        "expected_revision": 3,
        "responses": {"request-local": "yes"},
        **human_receipt,
        "confirmed": True,
    })
    assert list(task_update.iter_errors({
        "expected_revision": 3,
        "responses": {"request-local": "yes"},
        "confirmed": True,
    }))

    input_required_projection = _validator_for("mcpInputRequiredProjection")
    projected_input = {
        "kind": "input_required",
        "continuation_id": "continuation-local",
        "revision": 4,
        "respondable": True,
        "input_requests": [{
            "request_id": "request-local",
            "kind": "elicitation",
            "mode": "url",
            "schema": {"type": "object"},
            "inert_url": "https://provider.invalid/review",
        }],
        "human_request_id": "human-local",
        "human_revision": 2,
        "human_preview_sha256": "c" * 64,
    }
    input_required_projection.validate(projected_input)
    assert list(input_required_projection.iter_errors({
        **projected_input,
        "human_request_id": None,
    }))

    task_projection = _validator_for("mcpRemoteTaskProjection")
    task_projection.validate({
        "kind": "remote_task",
        "task_ref": "task-local",
        "revision": 1,
        "status": "working",
        "input_requests": [],
    })
    task_projection.validate({
        "kind": "remote_task",
        "task_ref": "task-local",
        "revision": 2,
        "status": "input_required",
        "input_requests": projected_input["input_requests"],
        "human_request_id": "human-task-local",
        "human_revision": 3,
        "human_preview_sha256": "d" * 64,
    })
    assert list(task_projection.iter_errors({
        "kind": "remote_task",
        "task_ref": "task-local",
        "revision": 2,
        "status": "input_required",
        "input_requests": projected_input["input_requests"],
    }))

    task_get = _validator_for("mcpRevisionReadPayload")
    task_get.validate({"expected_revision": 0})
    assert list(task_get.iter_errors({"expected_revision": -1}))

    subscription = _validator_for("mcpSubscriptionStartPayload")
    subscription.validate({"filters": ["resources/updated"], "confirmed": True})
    assert list(subscription.iter_errors({"filters": ["resources/updated", "resources/updated"], "confirmed": True}))

    schema = _schema()
    operations = schema["x-agent-libos-contract-scope"]["mcp_v3_host_operations"]
    assert operations["auth.status"]["route"].startswith("GET ")
    assert operations["auth.status"]["local_only"] is True
    assert all(
        item["route"].startswith("POST ")
        for name, item in operations.items()
        if name not in {"auth.status", "auth.profiles.list"}
    )
    for item in operations.values():
        request_def = item["request_def"]
        if request_def is not None:
            assert request_def in schema["$defs"]
        response_def = item.get("response_def")
        if response_def is not None:
            assert response_def in schema["$defs"]


def test_gui_api_schema_models_exact_non_secret_oauth_host_profile_admin() -> None:
    profile = {
        "profile_id": "profile-local",
        "server_id": "server-local",
        "resource_uri": "https://resource.example/mcp",
        "expected_issuer": "https://issuer.example",
        "redirect_uri": "http://127.0.0.1/callback",
        "client_id": "gui-client",
        "registration_mode": "preregistered",
        "token_endpoint_auth_method": "client_secret_basic",
        "allowed_scopes": ["resource.read"],
        "default_scopes": ["resource.read"],
        "allowed_endpoint_origins": ["https://issuer.example"],
        "allow_loopback_http": True,
        "protocol_revision": "2026-07-28",
        "transport": "streamable_http",
    }
    profile_validator = _validator_for("mcpOAuthProfileInput")
    profile_validator.validate(profile)
    for invalid in (
        {**profile, "registration_mode": "dcr"},
        {**profile, "client_secret": "must-not-enter-profile"},
        {key: value for key, value in profile.items() if key != "expected_issuer"},
        {**profile, "allow_loopback_http": "true"},
    ):
        assert list(profile_validator.iter_errors(invalid))

    mutation_validator = _validator_for("mcpOAuthProfileMutationPayload")
    mutation_validator.validate({
        "profile": profile,
        "client_secret": "one-time-secret",
        "replace": False,
        "confirmed": True,
    })
    for invalid in (
        {"profile": profile, "client_secret": None, "replace": False, "confirmed": True},
        {"profile": profile, "replace": False, "confirmed": False},
        {"profile": profile, "replace": False, "confirmed": True, "actor": "pid"},
        {"profile": profile, "replace": False, "confirmed": True, "client_secret": "bad\nsecret"},
    ):
        assert list(mutation_validator.iter_errors(invalid))

    status_list_validator = _validator_for("mcpOAuthStatusListProjection")
    status_list_validator.validate([{
        "profile_id": "profile-local",
        "status": "authorization_required",
        "scopes": [],
    }])
    assert list(status_list_validator.iter_errors([{
        "profile_id": "profile-local",
        "status": "authorized",
        "scopes": [],
        "access_token": "private",
    }]))

    operations = _schema()["x-agent-libos-contract-scope"]["mcp_v3_host_operations"]
    for name in ("auth.profile.add", "auth.profile.replace", "auth.profile.remove"):
        assert operations[name]["host_only"] is True
