from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import Context, Extension, MCPServer
from mcp.server.subscriptions import InMemorySubscriptionBus, ResourceUpdated
from mcp.types import (
    JSONRPCNotification,
    RequestParams,
    SubscriptionFilter,
    SubscriptionsAcknowledgedNotification,
    SubscriptionsAcknowledgedNotificationParams,
)
from pydantic import ConfigDict, Field


RESOURCE_URI = "fixture://document/current"
_document_revision = 1


def _task_state_path() -> Path | None:
    arguments = sys.argv[1:]
    if not arguments:
        return None
    if len(arguments) != 2 or arguments[0] != "--task-state-file":
        raise RuntimeError("invalid Python MCP fixture arguments")
    return Path(arguments[1]).resolve()


_TASK_STATE_PATH = _task_state_path()
_TASK_NOTIFICATION_SECRET = os.environ.get("MCP_FIXTURE_TASK_SECRET", "")


def _load_task_state() -> tuple[int, dict[str, dict[str, str]]]:
    if _TASK_STATE_PATH is None or not _TASK_STATE_PATH.exists():
        return 0, {}
    value = json.loads(_TASK_STATE_PATH.read_text(encoding="utf-8"))
    if (
        type(value) is not dict
        or type(value.get("counter")) is not int
        or type(value.get("tasks")) is not dict
    ):
        raise RuntimeError("invalid Python MCP fixture Task state")
    tasks = value["tasks"]
    if any(
        type(task_id) is not str
        or type(state) is not dict
        or set(state) != {"mode", "status"}
        or any(type(item) is not str for item in state.values())
        for task_id, state in tasks.items()
    ):
        raise RuntimeError("invalid Python MCP fixture Task row")
    return value["counter"], tasks


_task_counter, _tasks = _load_task_state()


def _save_task_state() -> None:
    if _TASK_STATE_PATH is None:
        return
    _TASK_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = _TASK_STATE_PATH.with_suffix(_TASK_STATE_PATH.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {"counter": _task_counter, "tasks": _tasks},
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    temporary.replace(_TASK_STATE_PATH)

_TASK_CREATED_AT = "2030-01-01T00:00:00Z"
_TASK_INITIAL_UPDATED_AT = "2030-01-01T00:00:01Z"
_TASK_INPUT_UPDATED_AT = "2030-01-01T00:00:02Z"
_TASK_COMPLETE_UPDATED_AT = "2030-01-01T00:00:03Z"


class _TaskReferenceParams(RequestParams):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    task_id: str = Field(alias="taskId")


class _TaskUpdateParams(_TaskReferenceParams):
    input_responses: dict[str, Any] = Field(alias="inputResponses")


class _FixtureSubscriptionBus(InMemorySubscriptionBus):
    """Emit one deterministic update only after a listen stream subscribes."""

    def subscribe(self, listener):  # type: ignore[no-untyped-def]
        unsubscribe = super().subscribe(listener)

        async def publish_fixture_update() -> None:
            await asyncio.sleep(0.2)
            await self.publish(ResourceUpdated(uri=RESOURCE_URI))

        asyncio.get_running_loop().create_task(
            publish_fixture_update(),
            name="mcp-python-v2-fixture-resource-update",
        )
        return unsubscribe


class _TasksExtension(Extension):
    identifier = "io.modelcontextprotocol/tasks"


server = MCPServer(
    "agent-libos-python-sdk-v2-fixture",
    version="2.0.0",
    subscriptions=_FixtureSubscriptionBus(),
    extensions=(_TasksExtension(),),
)


@server.resource(
    RESOURCE_URI,
    name="current-document",
    title="Current fixture document",
    description="A deterministic text resource used by the Agent libOS MCP gate.",
    mime_type="text/plain",
)
def current_document() -> str:
    return f"python-sdk-v2 revision={_document_revision}"


@server.resource(
    "fixture://document/{name}",
    name="named-document",
    title="Named fixture document",
    description="A deterministic resource-template fixture.",
    mime_type="text/plain",
)
def named_document(name: str) -> str:
    return f"python-sdk-v2 name={name}"


@server.prompt(
    name="review_document",
    title="Review fixture document",
    description="Build a deterministic review prompt.",
)
def review_document(focus: str = "correctness") -> str:
    return f"Review the fixture document for {focus}."


@server.tool(
    name="publish_resource_update",
    title="Publish fixture resource update",
    description="Increment the fixture resource and emit a modern subscription event.",
)
async def publish_resource_update(context: Context) -> dict[str, object]:
    global _document_revision
    _document_revision += 1
    await context.notify_resource_updated(RESOURCE_URI)
    return {"uri": RESOURCE_URI, "revision": _document_revision}


@server.tool(
    name="begin_review_task",
    title="Begin fixture review task",
    description="Return a digest-pinned Tasks extension handle.",
)
def begin_review_task(mode: str = "input") -> dict[str, str]:
    # The low-level exact-2026 extension handler below owns this result.  This
    # high-level registration exists only so tools/list advertises the same
    # independent SDK fixture Tool through the ordinary SDK catalog.
    return {"unreachable": mode}


def _task_result(task_id: str, state: dict[str, str], *, initial: bool) -> dict[str, Any]:
    status = state["status"]
    updated_at = {
        "working": _TASK_INITIAL_UPDATED_AT,
        "input_required": _TASK_INPUT_UPDATED_AT,
        "completed": _TASK_COMPLETE_UPDATED_AT,
        "cancelled": _TASK_COMPLETE_UPDATED_AT,
    }[status]
    result: dict[str, Any] = {
        "resultType": "task" if initial else "complete",
        "taskId": task_id,
        "status": status,
        "createdAt": _TASK_CREATED_AT,
        "lastUpdatedAt": updated_at,
        "ttlMs": 60_000,
        "pollIntervalMs": 0,
    }
    if status == "input_required":
        result["inputRequests"] = {
            "remote-review-input": {
                "method": "elicitation/create",
                "params": {
                    "message": "Approve the fixture review?",
                    "requestedSchema": {
                        "type": "object",
                        "properties": {"approved": {"type": "boolean"}},
                        "required": ["approved"],
                    },
                },
            }
        }
    elif status == "completed":
        result["result"] = {
            "approved": True,
            "source": "python-sdk-v2-tasks-extension",
        }
    return result


_lowlevel = server._lowlevel_server
_ordinary_tool_call = _lowlevel.get_request_handler("tools/call")
assert _ordinary_tool_call is not None
_ordinary_subscription_listen = _lowlevel.get_request_handler(
    "subscriptions/listen"
)
assert _ordinary_subscription_listen is not None


async def _handle_tool_call(context: Any, params: Any) -> dict[str, Any]:
    global _task_counter
    if params.name != "begin_review_task":
        return await _ordinary_tool_call.handler(context, params)
    arguments = params.arguments or {}
    mode = arguments.get("mode", "input")
    if mode not in {"input", "cancel"} or set(arguments) - {"mode"}:
        raise ValueError("invalid fixture Tasks mode")
    _task_counter += 1
    task_id = f"python-sdk-v2-private-task-{_task_counter}"
    _tasks[task_id] = {"mode": mode, "status": "working"}
    _save_task_state()
    return _task_result(task_id, _tasks[task_id], initial=True)


async def _handle_subscription_listen(context: Any, params: Any) -> Any:
    requested = params.notifications.model_dump(
        by_alias=True,
        mode="json",
        exclude_none=True,
    )
    task_ids = requested.get("taskIds")
    if task_ids is None:
        return await _ordinary_subscription_listen.handler(context, params)
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or any(type(task_id) is not str or task_id not in _tasks for task_id in task_ids)
        or len(set(task_ids)) != len(task_ids)
    ):
        raise ValueError("invalid fixture Task subscription")
    subscription_id = context.request_id
    if subscription_id is None:
        raise ValueError("fixture Task subscription requires a request id")
    metadata = {
        "io.modelcontextprotocol/subscriptionId": subscription_id,
    }
    honored = SubscriptionFilter.model_validate({"taskIds": task_ids})
    await context.session.send_notification(
        SubscriptionsAcknowledgedNotification(
            params=SubscriptionsAcknowledgedNotificationParams(
                notifications=honored,
                _meta=metadata,
            )
        ),
        related_request_id=subscription_id,
    )
    remote_id = task_ids[0]
    await context.session.send_notification(
        JSONRPCNotification(
            jsonrpc="2.0",
            method="notifications/tasks/status",
            params={
                "taskId": remote_id,
                "status": _tasks[remote_id]["status"],
                "statusMessage": f"fixture update {_TASK_NOTIFICATION_SECRET}",
                "createdAt": _TASK_CREATED_AT,
                "lastUpdatedAt": _TASK_INITIAL_UPDATED_AT,
                "ttlMs": 60_000,
                "pollIntervalMs": 0,
                "ui/resourceUri": f"ui://fixture/{_TASK_NOTIFICATION_SECRET}",
                "ui/visibility": ["model"],
                "_meta": metadata,
            },
        ),
        related_request_id=subscription_id,
    )
    await asyncio.Future()


async def _handle_task_get(
    _context: Any,
    params: _TaskReferenceParams,
) -> dict[str, Any]:
    state = _tasks.get(params.task_id)
    if state is None:
        raise ValueError("unknown fixture Task")
    if state["mode"] == "input" and state["status"] == "working":
        state["status"] = "input_required"
        _save_task_state()
    return _task_result(params.task_id, state, initial=False)


async def _handle_task_update(
    _context: Any,
    params: _TaskUpdateParams,
) -> dict[str, Any]:
    state = _tasks.get(params.task_id)
    if state is None or state["status"] != "input_required":
        raise ValueError("fixture Task is not awaiting input")
    if params.input_responses != {
        "remote-review-input": {
            "action": "accept",
            "content": {"approved": True},
        }
    }:
        raise ValueError("fixture Task received an unbound input response")
    state["status"] = "completed"
    _save_task_state()
    return {}


async def _handle_task_cancel(
    _context: Any,
    params: _TaskReferenceParams,
) -> dict[str, Any]:
    state = _tasks.get(params.task_id)
    if state is None or state["status"] not in {"working", "input_required"}:
        raise ValueError("fixture Task cannot be cancelled")
    state["status"] = "cancelled"
    _save_task_state()
    return {}


_lowlevel.add_request_handler(
    "tools/call",
    _ordinary_tool_call.params_type,
    _handle_tool_call,
)
_lowlevel.add_request_handler("tasks/get", _TaskReferenceParams, _handle_task_get)
_lowlevel.add_request_handler("tasks/update", _TaskUpdateParams, _handle_task_update)
_lowlevel.add_request_handler("tasks/cancel", _TaskReferenceParams, _handle_task_cancel)
_lowlevel.add_request_handler(
    "subscriptions/listen",
    _ordinary_subscription_listen.params_type,
    _handle_subscription_listen,
)


if __name__ == "__main__":
    server.run("stdio")
