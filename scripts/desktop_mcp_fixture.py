from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import Context, Extension, MCPServer, RequestStateSecurity
from mcp.types import (
    Completion,
    ElicitRequest,
    ElicitRequestFormParams,
    InputRequiredResult,
    PromptReference,
    RequestParams,
)
from pydantic import ConfigDict, Field


RESOURCE_URI = "desktop://status"
TEMPLATE_URI = "desktop://greeting/{name}"
MRTR_STATE = "desktop-frozen-mrtr-private-state"
TASK_CREATED_AT = "2030-01-01T00:00:00Z"
TASK_WORKING_AT = "2030-01-01T00:00:01Z"
TASK_INPUT_AT = "2030-01-01T00:00:02Z"
TASK_COMPLETE_AT = "2030-01-01T00:00:03Z"


class TasksExtension(Extension):
    identifier = "io.modelcontextprotocol/tasks"


class TaskReferenceParams(RequestParams):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    task_id: str = Field(alias="taskId")


class TaskUpdateParams(TaskReferenceParams):
    input_responses: dict[str, Any] = Field(alias="inputResponses")


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"counter": 0, "tasks": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        type(value) is not dict
        or type(value.get("counter")) is not int
        or type(value.get("tasks")) is not dict
    ):
        raise RuntimeError("desktop MCP fixture state is invalid")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-state-file", type=Path, required=True)
    args = parser.parse_args()
    state_path = args.task_state_file.resolve()
    state = _load_state(state_path)
    server = MCPServer(
        "agent-libos-desktop-frozen-smoke",
        version="2.0.0",
        extensions=(TasksExtension(),),
        # Agent libOS intentionally opens a fresh governed stdio process for
        # each stateless operation.  Use one deterministic fixture-only key so
        # a sealed MRTR requestState survives that process boundary, just as a
        # production multi-instance MCP server would use a shared key ring.
        request_state_security=RequestStateSecurity(
            keys=[hashlib.sha256(b"agent-libos-desktop-mrtr-fixture").digest()],
        ),
        log_level="ERROR",
    )

    def save_state() -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(state, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(state_path)

    @server.resource(
        RESOURCE_URI,
        name="desktop-status",
        description="Frozen desktop Resource smoke.",
        mime_type="text/plain",
    )
    def status() -> str:
        return "frozen desktop resource ready"

    @server.resource(
        TEMPLATE_URI,
        name="desktop-greeting",
        description="Frozen desktop Resource Template smoke.",
        mime_type="text/plain",
    )
    def greeting(name: str) -> str:
        return f"frozen desktop says hello to {name}"

    @server.prompt(
        name="desktop_review",
        description="Frozen desktop Prompt smoke.",
    )
    def prompt(subject: str) -> str:
        return f"Review frozen desktop subject {subject}."

    @server.completion()
    async def complete(reference: object, argument: object, context: object) -> Completion:
        del context
        if (
            isinstance(reference, PromptReference)
            and reference.name == "desktop_review"
            and getattr(argument, "name", None) == "subject"
        ):
            return Completion(
                values=[f"{getattr(argument, 'value', '')}-frozen"],
                total=1,
                hasMore=False,
            )
        return Completion(values=[], total=0, hasMore=False)

    @server.tool(name="desktop_echo", description="Frozen desktop Tool smoke.")
    def echo(text: str) -> dict[str, str]:
        return {"echo": text, "runtime": "frozen-desktop"}

    @server.tool(name="desktop_review_mrtr", description="Frozen desktop MRTR smoke.")
    def review_mrtr(document: str, context: Context) -> dict[str, Any] | InputRequiredResult:
        responses = context.input_responses
        if responses is None:
            return InputRequiredResult(
                inputRequests={
                    "desktop-review-input": ElicitRequest(
                        params=ElicitRequestFormParams(
                            message="Approve the frozen desktop review?",
                            requestedSchema={
                                "type": "object",
                                "properties": {"approved": {"type": "boolean"}},
                                "required": ["approved"],
                            },
                        )
                    )
                },
                requestState=MRTR_STATE,
            )
        response = responses["desktop-review-input"]
        approved = bool(response.content and response.content.get("approved"))
        if context.request_state != MRTR_STATE or not approved:
            raise ValueError("frozen desktop MRTR answer is not bound")
        return {"document": document, "approved": approved, "rounds": 2}

    @server.tool(name="desktop_begin_task", description="Frozen desktop Task smoke.")
    def begin_task(mode: str = "input") -> dict[str, str]:
        return {"unreachable": mode}

    def task_result(task_id: str, *, initial: bool) -> dict[str, Any]:
        selected = state["tasks"][task_id]
        status_value = selected["status"]
        updated_at = {
            "working": TASK_WORKING_AT,
            "input_required": TASK_INPUT_AT,
            "completed": TASK_COMPLETE_AT,
            "cancelled": TASK_COMPLETE_AT,
        }[status_value]
        result: dict[str, Any] = {
            "resultType": "task" if initial else "complete",
            "taskId": task_id,
            "status": status_value,
            "createdAt": TASK_CREATED_AT,
            "lastUpdatedAt": updated_at,
            "ttlMs": 60_000,
            "pollIntervalMs": 0,
        }
        if status_value == "input_required":
            result["inputRequests"] = {
                "desktop-task-input": {
                    "method": "elicitation/create",
                    "params": {
                        "mode": "form",
                        "message": "Approve the frozen desktop Task?",
                        "requestedSchema": {
                            "type": "object",
                            "properties": {"approved": {"type": "boolean"}},
                            "required": ["approved"],
                        },
                    },
                }
            }
        elif status_value == "completed":
            result["result"] = {"approved": True, "runtime": "frozen-desktop"}
        return result

    lowlevel = server._lowlevel_server
    ordinary_call = lowlevel.get_request_handler("tools/call")
    if ordinary_call is None:
        raise RuntimeError("desktop MCP fixture has no tools/call handler")

    async def call_tool(context: Any, params: Any) -> Any:
        if params.name != "desktop_begin_task":
            return await ordinary_call.handler(context, params)
        arguments = params.arguments or {}
        if set(arguments) - {"mode"} or arguments.get("mode", "input") != "input":
            raise ValueError("desktop Task fixture arguments are invalid")
        state["counter"] += 1
        task_id = f"desktop-private-task-{state['counter']}"
        state["tasks"][task_id] = {"status": "working"}
        save_state()
        return task_result(task_id, initial=True)

    async def task_get(_context: Any, params: TaskReferenceParams) -> dict[str, Any]:
        selected = state["tasks"].get(params.task_id)
        if selected is None:
            raise ValueError("unknown desktop Task")
        if selected["status"] == "working":
            selected["status"] = "input_required"
            save_state()
        return task_result(params.task_id, initial=False)

    async def task_update(_context: Any, params: TaskUpdateParams) -> dict[str, Any]:
        selected = state["tasks"].get(params.task_id)
        if selected is None or selected["status"] != "input_required":
            raise ValueError("desktop Task is not awaiting input")
        expected = {
            "desktop-task-input": {
                "action": "accept",
                "content": {"approved": True},
            }
        }
        if params.input_responses != expected:
            raise ValueError("desktop Task answer is not bound")
        selected["status"] = "completed"
        save_state()
        return {}

    async def task_cancel(_context: Any, params: TaskReferenceParams) -> dict[str, Any]:
        selected = state["tasks"].get(params.task_id)
        if selected is None or selected["status"] not in {"working", "input_required"}:
            raise ValueError("desktop Task cannot be cancelled")
        selected["status"] = "cancelled"
        save_state()
        return {}

    lowlevel.add_request_handler("tools/call", ordinary_call.params_type, call_tool)
    lowlevel.add_request_handler("tasks/get", TaskReferenceParams, task_get)
    lowlevel.add_request_handler("tasks/update", TaskUpdateParams, task_update)
    lowlevel.add_request_handler("tasks/cancel", TaskReferenceParams, task_cancel)
    server.run("stdio")


if __name__ == "__main__":
    main()
