from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import CapabilityRight, CapabilitySpec, ObjectRight, ProcessMessageKind, ProcessSignal, ProcessStatus
from agent_libos.models.exceptions import (
    CapabilityDenied,
    HumanApprovalRequired,
    NotFound,
    ProcessMessageWaitRequired,
    ProcessWaitRequired,
    ValidationError,
)
from agent_libos.runtime.runtime import Runtime
from agent_libos.utils.serde import to_jsonable

_RUNTIME_DEFAULTS = DEFAULT_CONFIG.runtime
_GUI_DEFAULTS = DEFAULT_CONFIG.gui
_TERMINAL = {ProcessStatus.EXITED, ProcessStatus.FAILED, ProcessStatus.KILLED}
_CONFIG_DEFAULT = object()


def _bounded_gui_value(value: Any, *, string_limit: int, collection_limit: int) -> Any:
    jsonable = to_jsonable(value)
    if isinstance(jsonable, str):
        if len(jsonable) <= string_limit:
            return jsonable
        return {
            "truncated": True,
            "chars": len(jsonable),
            "preview": jsonable[:string_limit],
        }
    if isinstance(jsonable, list):
        selected = [_bounded_gui_value(item, string_limit=string_limit, collection_limit=collection_limit) for item in jsonable[:collection_limit]]
        if len(jsonable) > collection_limit:
            selected.append({"truncated": True, "omitted": len(jsonable) - collection_limit})
        return selected
    if isinstance(jsonable, dict):
        items = list(jsonable.items())
        bounded = {
            str(key): _bounded_gui_value(item, string_limit=string_limit, collection_limit=collection_limit)
            for key, item in items[:collection_limit]
        }
        if len(items) > collection_limit:
            bounded["_truncated"] = {"omitted": len(items) - collection_limit}
        return bounded
    return jsonable


def _sse_payload_data(data: dict[str, Any], *, max_bytes: int, string_limit: int, collection_limit: int) -> dict[str, Any]:
    bounded = _bounded_gui_value(data, string_limit=string_limit, collection_limit=collection_limit)
    encoded = json.dumps(bounded, ensure_ascii=False, default=str).encode("utf-8")
    if len(encoded) <= max_bytes:
        return bounded
    return {
        "truncated": True,
        "bytes": len(encoded),
        "reason": "gui event payload exceeds sse_payload_max_bytes",
    }


class GuiServerError(Exception):
    def __init__(self, status: int, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.details = details or {}


@dataclass
class GuiEvent:
    seq: int
    event: str
    data: dict[str, Any]


class GuiEventBroadcaster:
    """In-process event buffer used by the GUI SSE endpoint."""

    def __init__(self, max_events: int = _GUI_DEFAULTS.event_buffer_limit) -> None:
        self._condition = threading.Condition()
        self._events: list[GuiEvent] = []
        self._next_seq = 1
        self._max_events = max_events
        self._closed = False

    def publish(self, event: str, data: dict[str, Any] | None = None) -> GuiEvent:
        with self._condition:
            item = GuiEvent(seq=self._next_seq, event=event, data=data or {})
            self._next_seq += 1
            self._events.append(item)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events :]
            self._condition.notify_all()
            return item

    def replay_after(self, cursor: int) -> list[GuiEvent]:
        with self._condition:
            return [event for event in self._events if event.seq > cursor]

    def wait_after(self, cursor: int, timeout_s: float = 15.0) -> list[GuiEvent]:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while not self._closed:
                ready = [event for event in self._events if event.seq > cursor]
                if ready:
                    return ready
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self._condition.wait(remaining)
            return []

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


@dataclass
class SchedulerController:
    service: "GuiRuntimeService"
    auto_run: bool = True
    default_max_quanta: int | None = None
    running: bool = False
    paused: bool = False
    task_id: str | None = None
    reason: str | None = None
    last_result: list[Any] = field(default_factory=list)
    last_error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _thread: threading.Thread | None = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "auto_run": self.auto_run,
                "running": self.running,
                "paused": self.paused,
                "task_id": self.task_id,
                "reason": self.reason,
                "last_result": to_jsonable(self.last_result),
                "last_error": self.last_error,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "default_max_quanta": self.default_max_quanta,
            }

    def set_auto_run(self, enabled: bool) -> dict[str, Any]:
        with self._lock:
            self.auto_run = bool(enabled)
            self.paused = not self.auto_run
        self.service.publish_scheduler_status()
        return self.status()

    def pause(self) -> dict[str, Any]:
        with self._lock:
            self.paused = True
            self.auto_run = False
        self.service.publish_scheduler_status()
        return self.status()

    def maybe_start(self, *, max_quanta: int | None = None, reason: str = "auto") -> dict[str, Any]:
        if not self.auto_run or self.paused:
            return self.status()
        return self.start(max_quanta=max_quanta, reason=reason)

    def start(self, *, pid: str | None = None, max_quanta: int | None = None, reason: str = "manual") -> dict[str, Any]:
        with self._lock:
            if self.running:
                return self.status()
            self.running = True
            self.task_id = f"scheduler-{int(time.time() * 1000)}"
            self.reason = reason
            self.started_at = time.time()
            self.finished_at = None
            self.last_error = None
            selected_quanta = max_quanta if max_quanta is not None else self.default_max_quanta
            thread = threading.Thread(
                target=self._run_background,
                args=(selected_quanta, pid),
                name="agent-libos-gui-scheduler",
                daemon=True,
            )
            self._thread = thread
            thread.start()
        self.service.publish_scheduler_status()
        return self.status()

    def shutdown(self, timeout_s: float | None = None) -> bool:
        selected_timeout = self.service.runtime.config.gui.scheduler_shutdown_join_timeout_s if timeout_s is None else timeout_s
        with self._lock:
            self.paused = True
            self.auto_run = False
            thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=selected_timeout)
        with self._lock:
            if thread is None or not thread.is_alive():
                self.running = False
                self.finished_at = self.finished_at or time.time()
                return True
            return False

    def is_running(self) -> bool:
        with self._lock:
            thread = self._thread
            return self.running and thread is not None and thread.is_alive()

    def run_step(self, pid: str) -> dict[str, Any]:
        with self._lock:
            if self.running:
                return {"started": False, "scheduler": self.status()}
            self.running = True
            self.task_id = f"step-{int(time.time() * 1000)}"
            self.reason = f"step:{pid}"
            self.started_at = time.time()
            self.finished_at = None
            self.last_error = None
        self.service.publish_scheduler_status()
        try:
            with self.service.runtime_lock:
                result = asyncio.run(self.service.runtime.arun_process_once(pid))
                self.last_result = [result]
                self.service.publish_runtime_changes("step")
            return {"started": True, "result": to_jsonable(result), "scheduler": self.status()}
        except Exception as exc:
            self.last_error = str(exc)
            raise
        finally:
            with self._lock:
                self.running = False
                self.finished_at = time.time()
            self.service.publish_scheduler_status()

    def _run_background(self, max_quanta: int | None, pid: str | None) -> None:
        try:
            with self.service.runtime_lock:
                result = (
                    self.service.runtime.run_process_until_idle(pid, max_quanta=max_quanta)
                    if pid is not None
                    else self.service.runtime.run_until_idle(max_quanta=max_quanta)
                )
            with self._lock:
                self.last_result = result
        except Exception as exc:  # pragma: no cover - covered through API status assertions
            with self._lock:
                self.last_error = str(exc)
        finally:
            with self._lock:
                self.running = False
                self.finished_at = time.time()
            self.service.publish_runtime_changes("scheduler")
            self.service.publish_scheduler_status()


class GuiRuntimeService:
    """Local-only HTTP facade over one Agent libOS Runtime instance."""

    def __init__(
        self,
        *,
        db: str = _RUNTIME_DEFAULTS.local_store_target,
        runtime: Runtime | None = None,
        token: str | None = None,
        auto_run: bool = True,
        max_quanta: int | None | object = _CONFIG_DEFAULT,
    ) -> None:
        self.db = db
        self.runtime = runtime or Runtime.open(db)
        self.owns_runtime = runtime is None
        self.token = token or secrets.token_urlsafe(32)
        self.broadcaster = GuiEventBroadcaster(max_events=self.runtime.config.gui.event_buffer_limit)
        self.runtime_lock = threading.RLock()
        selected_max_quanta = self.runtime.config.runtime.run_until_idle_max_quanta if max_quanta is _CONFIG_DEFAULT else max_quanta
        self.scheduler = SchedulerController(self, auto_run=auto_run, default_max_quanta=selected_max_quanta, paused=not auto_run)
        self._closed = False
        self._static_snapshot_cache: dict[str, Any] | None = None
        self._static_snapshot_dirty = True
        self._seen_event_ids: set[str] = set()
        self._seen_audit_ids: set[str] = set()
        self._seen_human_request_ids: set[str] = set()
        self._seen_message_ids: set[str] = set()
        self._seen_llm_call_ids: set[str] = set()
        self.publish_runtime_changes("startup")

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        scheduler_stopped = self.scheduler.shutdown()
        self.broadcaster.close()
        # A Python thread blocked inside a model/tool quantum cannot be safely
        # interrupted. In that case the Electron parent will terminate the
        # process tree; closing SQLite underneath the live quantum would be a
        # worse race than letting process teardown reclaim the handle.
        if self.owns_runtime and scheduler_stopped:
            self.runtime.shutdown(actor="gui-server", reason="gui-server.shutdown")

    def close(self) -> None:
        self.shutdown()

    def publish_scheduler_status(self) -> None:
        self.broadcaster.publish("scheduler.status", self.scheduler.status())

    def publish_runtime_changes(self, reason: str) -> None:
        with self.runtime_lock:
            if self._reason_changes_static_snapshot(reason):
                self._static_snapshot_dirty = True
            snapshot = self.snapshot()
            self.broadcaster.publish("snapshot", {"reason": reason, "snapshot": snapshot})
            for event in snapshot["events"]:
                if event["event_id"] in self._seen_event_ids:
                    continue
                self._seen_event_ids.add(event["event_id"])
                self.broadcaster.publish("event.appended", event)
            for record in snapshot["audit"]:
                if record["record_id"] in self._seen_audit_ids:
                    continue
                self._seen_audit_ids.add(record["record_id"])
                self.broadcaster.publish("audit.appended", record)
            for request in snapshot["human_requests"]:
                if request["request_id"] in self._seen_human_request_ids:
                    continue
                self._seen_human_request_ids.add(request["request_id"])
                self.broadcaster.publish("human_request.updated", request)
            for process in snapshot["processes"]:
                for message in process.get("messages", []):
                    if message["message_id"] in self._seen_message_ids:
                        continue
                    self._seen_message_ids.add(message["message_id"])
                    self.broadcaster.publish("message.posted", message)
            for call in snapshot["llm_calls"]:
                if call["call_id"] in self._seen_llm_call_ids:
                    continue
                self._seen_llm_call_ids.add(call["call_id"])
                self.broadcaster.publish("llm_call.appended", call)

    def health(self) -> dict[str, Any]:
        with self.runtime_lock:
            return {
                "ok": True,
                "db": self.db,
                "scheduler": self.scheduler.status(),
                "process_count": len(self.runtime.process.list()),
            }

    def snapshot(self) -> dict[str, Any]:
        with self.runtime_lock:
            processes = [self._process_summary(process.pid, include_messages=True) for process in self.runtime.process.list()]
            static = self._static_snapshot()
            snapshot = {
                "db": self.db,
                "scheduler": self.scheduler.status(),
                "processes": processes,
                "human_requests": to_jsonable(self.runtime.human.list()),
                "events": to_jsonable(self.runtime.events.list()[-self.runtime.config.gui.snapshot_event_limit :]),
                "audit": to_jsonable(self.runtime.audit.trace(limit=self.runtime.config.gui.snapshot_audit_limit)),
                "llm_calls": to_jsonable(self.runtime.store.list_llm_calls(limit=self.runtime.config.gui.snapshot_llm_call_limit)),
                "object_tasks": to_jsonable(self.runtime.object_tasks.list(limit=self.runtime.config.gui.snapshot_object_task_limit)),
                **static,
            }
            return self._bounded_snapshot(snapshot)

    def _static_snapshot(self) -> dict[str, Any]:
        if self._static_snapshot_cache is None or self._static_snapshot_dirty:
            self._static_snapshot_cache = {
                "tools": self._tool_summaries(),
                "images": to_jsonable(self.runtime.image_registry.list_images()),
                "skills": to_jsonable(self.runtime.skills.discover_skills(require_capability=False)),
                "jsonrpc_endpoints": to_jsonable(self.runtime.jsonrpc.list_endpoints(require_capability=False)),
                "modules": to_jsonable(self.runtime.modules.loaded_module_summaries()),
            }
            self._static_snapshot_dirty = False
        return dict(self._static_snapshot_cache)

    def _reason_changes_static_snapshot(self, reason: str) -> bool:
        return reason.startswith(("image.", "skill.", "jsonrpc.", "module.", "process.exec"))

    def _process_summary(self, pid: str, *, include_messages: bool = False) -> dict[str, Any]:
        process = self.runtime.process.get(pid)
        unread = self.runtime.messages.list(pid, include_acked=False)
        messages = (
            self.runtime.messages.list(pid, include_acked=True, limit=self.runtime.config.gui.snapshot_process_message_limit)
            if include_messages
            else []
        )
        calls = self.runtime.store.list_llm_calls(pid=pid, limit=self.runtime.config.gui.snapshot_process_llm_call_limit)
        return {
            **to_jsonable(process),
            "terminal": process.status in _TERMINAL,
            "unread_message_count": len(unread),
            "interrupt_count": len([item for item in unread if item.kind == ProcessMessageKind.INTERRUPT]),
            "messages": to_jsonable(messages),
            "llm_call_count": len(calls),
            "token_total": sum(int((call.usage or {}).get("total_tokens", 0) or 0) for call in calls),
            "resource_remaining": to_jsonable(self.runtime.resources.remaining_budget(pid)),
        }

    def _bounded_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        return _bounded_gui_value(
            snapshot,
            string_limit=self.runtime.config.gui.snapshot_string_max_chars,
            collection_limit=self.runtime.config.gui.snapshot_collection_max_items,
        )

    def _tool_summaries(self) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for tool in self.runtime.tools.list():
            try:
                spec = json.loads(tool.get("spec_json") or "{}")
            except json.JSONDecodeError:
                spec = {}
            summaries.append(
                {
                    "tool_id": tool.get("tool_id"),
                    "name": tool.get("name"),
                    "scope": tool.get("scope"),
                    "registered_by": tool.get("registered_by"),
                    "ephemeral": bool(tool.get("ephemeral")),
                    "description": spec.get("description", ""),
                    "tags": spec.get("tags", []),
                    "policy": spec.get("policy", {}),
                }
            )
        return summaries


class GuiHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], service: GuiRuntimeService):
        super().__init__(server_address, GuiRequestHandler)
        self.service = service


class GuiRequestHandler(BaseHTTPRequestHandler):
    server: GuiHTTPServer

    def handle(self) -> None:
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_common_headers()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._handle("POST")

    def _handle(self, method: str) -> None:
        parsed = urlparse(self.path)
        self._body_cached = False
        self._cached_json_body: dict[str, Any] = {}
        try:
            self._require_auth()
            if method == "GET" and parsed.path == "/api/events/stream":
                self._handle_sse(parsed)
                return
            if method == "POST":
                self._cached_json_body = self._read_body(optional=True)
                self._body_cached = True
            if _is_object_task_wait_request(method, parsed.path):
                result = self._dispatch(method, parsed.path, parse_qs(parsed.query))
                should_shutdown = method == "POST" and parsed.path == "/api/shutdown"
            else:
                with self.server.service.runtime_lock:
                    result = self._dispatch(method, parsed.path, parse_qs(parsed.query))
                    should_shutdown = method == "POST" and parsed.path == "/api/shutdown"
            if should_shutdown:
                self.close_connection = True
            self._write_json(result)
            if should_shutdown:
                self._schedule_server_shutdown()
        except GuiServerError as exc:
            self._write_json({"ok": False, "error": {"message": str(exc), **exc.details}}, status=exc.status)
        except CapabilityDenied as exc:
            self._write_json(
                {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}},
                status=HTTPStatus.FORBIDDEN,
            )
        except HumanApprovalRequired as exc:
            self._write_json(
                {
                    "ok": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "request_id": exc.request_id,
                    },
                },
                status=HTTPStatus.CONFLICT,
            )
        except ProcessWaitRequired as exc:
            self._write_json(
                {
                    "ok": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "child_pid": exc.child_pid,
                    },
                },
                status=HTTPStatus.CONFLICT,
            )
        except ProcessMessageWaitRequired as exc:
            self._write_json(
                {
                    "ok": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "recipient_pid": exc.recipient_pid,
                        "filters": exc.filters,
                    },
                },
                status=HTTPStatus.CONFLICT,
            )
        except NotFound as exc:
            self._write_json(
                {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}},
                status=HTTPStatus.NOT_FOUND,
            )
        except ValidationError as exc:
            self._write_json(
                {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}},
                status=HTTPStatus.BAD_REQUEST,
            )
        except Exception as exc:
            self._write_json(
                {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _dispatch(self, method: str, path: str, query: dict[str, list[str]]) -> Any:
        service = self.server.service
        parts = [unquote(part) for part in path.strip("/").split("/") if part]
        if parts[:1] != ["api"]:
            raise GuiServerError(HTTPStatus.NOT_FOUND, "unknown endpoint")
        route = parts[1:]
        if method == "GET" and route == ["health"]:
            return service.health()
        if method == "POST" and route == ["shutdown"]:
            service.scheduler.pause()
            return {"ok": True, "status": "shutting_down"}
        if method == "GET" and route == ["snapshot"]:
            return service.snapshot()
        if method == "GET" and route == ["processes"]:
            return [service._process_summary(process.pid, include_messages=True) for process in service.runtime.process.list()]
        if method == "GET" and route == ["tools"]:
            return service._tool_summaries()
        if method == "POST" and route == ["processes"]:
            body = self._read_body()
            pid = service.runtime.process.spawn(
                image=str(body["image"]) if body.get("image") is not None else None,
                goal=body.get("goal", ""),
                working_directory=body.get("working_directory"),
                llm_profile_id=str(body["llm_profile"]) if body.get("llm_profile") is not None else None,
            )
            service.publish_runtime_changes("process.spawn")
            if body.get("auto_run", True):
                service.scheduler.maybe_start(
                    max_quanta=_positive_int_or_none(body.get("max_quanta"), "max_quanta"),
                    reason=f"spawn:{pid}",
                )
            return {"pid": pid, "process": service._process_summary(pid, include_messages=True), "scheduler": service.scheduler.status()}
        if len(route) >= 1 and route[0] == "workflows":
            return self._dispatch_workflows(method, route[1:])
        if len(route) >= 1 and route[0] == "object-tasks":
            return self._dispatch_object_tasks(method, route[1:], query)
        if route[:2] == ["scheduler", "auto"] and method == "POST":
            body = self._read_body()
            return service.scheduler.set_auto_run(bool(body.get("enabled", True)))
        if route == ["scheduler", "pause"] and method == "POST":
            return service.scheduler.pause()
        if len(route) >= 2 and route[0] == "processes":
            return self._dispatch_process(method, route[1], route[2:], query)
        if len(route) >= 1 and route[0] == "human-requests":
            return self._dispatch_human(method, route[1:])
        if len(route) >= 1 and route[0] == "checkpoints":
            return self._dispatch_checkpoints(method, route[1:], query)
        if len(route) >= 1 and route[0] == "skills":
            return self._dispatch_skills(method, route[1:], query)
        if len(route) >= 1 and route[0] == "capabilities":
            return self._dispatch_capabilities(method, route[1:], query)
        if len(route) >= 1 and route[0] == "images":
            return self._dispatch_images(method, route[1:])
        if len(route) >= 1 and route[0] == "jsonrpc":
            return self._dispatch_jsonrpc(method, route[1:], query)
        if len(route) >= 1 and route[0] == "modules":
            return self._dispatch_modules(method, route[1:])
        raise GuiServerError(HTTPStatus.NOT_FOUND, "unknown endpoint")

    def _dispatch_workflows(self, method: str, route: list[str]) -> Any:
        service = self.server.service
        if method == "POST" and route == ["run"]:
            body = self._read_body()
            tool = str(body.get("tool") or "").strip()
            if not tool:
                raise GuiServerError(HTTPStatus.BAD_REQUEST, "workflow tool is required")
            raw_args = body.get("args") if "args" in body else {}
            if not isinstance(raw_args, dict):
                raise GuiServerError(HTTPStatus.BAD_REQUEST, "workflow args must be a JSON object")
            result = service.runtime.run_workflow(
                tool,
                raw_args,
                image=str(body["image"]) if body.get("image") is not None else None,
                goal=body.get("goal"),
                working_directory=str(body["working_directory"]) if body.get("working_directory") is not None else None,
            )
            service.publish_runtime_changes("workflow.run")
            return to_jsonable(result)
        raise GuiServerError(HTTPStatus.NOT_FOUND, "unknown workflows endpoint")

    def _dispatch_object_tasks(self, method: str, route: list[str], query: dict[str, list[str]]) -> Any:
        service = self.server.service
        if method == "GET" and not route:
            return to_jsonable(
                service.runtime.object_tasks.list(
                    actor_pid=_query_str(query, "pid"),
                    owner_oid=_query_str(query, "owner_oid"),
                    include_terminal=_query_str(query, "active") not in {"1", "true", "yes"},
                    limit=_query_int(query, "limit"),
                )
            )
        if method == "POST" and route == ["start"]:
            body = self._read_body()
            pid = str(body.get("pid") or "").strip()
            if not pid:
                raise GuiServerError(HTTPStatus.BAD_REQUEST, "pid is required")
            tool = str(body.get("tool") or "").strip()
            if not tool:
                raise GuiServerError(HTTPStatus.BAD_REQUEST, "tool is required")
            raw_args = body.get("args") if "args" in body else {}
            if not isinstance(raw_args, dict):
                raise GuiServerError(HTTPStatus.BAD_REQUEST, "object task args must be a JSON object")
            owner = _object_task_owner_handle(
                service.runtime,
                pid,
                body.get("owner_oid"),
                body.get("owner_name"),
                body.get("namespace"),
            )
            task = service.runtime.object_tasks.start(
                pid,
                owner,
                tool,
                raw_args,
                notify_pid=str(body["notify_pid"]) if body.get("notify_pid") is not None else None,
                notify_kind=str(body.get("notify_kind") or ProcessMessageKind.NORMAL.value),
                notify_channel=str(body["notify_channel"]) if body.get("notify_channel") is not None else None,
                inherit_capabilities=body.get("inherit_capabilities") if isinstance(body.get("inherit_capabilities"), list) else [],
                grant_result_to_notify=bool(body.get("grant_result_to_notify", False)),
                owner_watch=_object_task_owner_watch_body(body),
            )
            service.publish_runtime_changes("object_task.start")
            return to_jsonable(task)
        if len(route) == 1 and method == "GET":
            return to_jsonable(service.runtime.object_tasks.get(route[0], actor_pid=_query_str(query, "pid")))
        if len(route) == 2 and route[1] == "cancel" and method == "POST":
            body = self._read_body()
            pid = str(body.get("pid") or "").strip()
            if not pid:
                raise GuiServerError(HTTPStatus.BAD_REQUEST, "pid is required")
            task = service.runtime.object_tasks.cancel(route[0], actor_pid=pid, reason=body.get("reason"))
            service.publish_runtime_changes("object_task.cancel")
            return to_jsonable(task)
        if len(route) == 2 and route[1] == "wait" and method == "POST":
            body = self._read_body(optional=True)
            pid = str(body.get("pid")) if body.get("pid") is not None else None
            task = service.runtime.object_tasks.wait(
                route[0],
                actor_pid=pid,
                timeout=_bounded_float_or_default(
                    body.get("timeout_s"),
                    "timeout_s",
                    default=service.runtime.config.gui.object_task_wait_default_timeout_s,
                    maximum=service.runtime.config.gui.object_task_wait_max_timeout_s,
                ),
            )
            service.publish_runtime_changes("object_task.wait")
            return to_jsonable(task)
        if len(route) == 2 and route[1] == "watch-owner" and method == "POST":
            body = self._read_body()
            pid = str(body.get("pid") or "").strip()
            if not pid:
                raise GuiServerError(HTTPStatus.BAD_REQUEST, "pid is required")
            raw_events = body.get("watch_events")
            if raw_events is not None and not isinstance(raw_events, list):
                raise GuiServerError(HTTPStatus.BAD_REQUEST, "watch_events must be a JSON array")
            task = service.runtime.object_tasks.watch_owner(
                route[0],
                actor_pid=pid,
                enabled=bool(body.get("enabled", True)),
                events=[str(item) for item in raw_events] if raw_events is not None else None,
                channel=str(body["watch_channel"]) if body.get("watch_channel") is not None else None,
                kind=str(body["watch_kind"]) if body.get("watch_kind") is not None else None,
            )
            service.publish_runtime_changes("object_task.watch_owner")
            return to_jsonable(task)
        raise GuiServerError(HTTPStatus.NOT_FOUND, "unknown object-tasks endpoint")

    def _dispatch_process(self, method: str, pid: str, route: list[str], query: dict[str, list[str]]) -> Any:
        service = self.server.service
        if method == "GET" and not route:
            return service._process_summary(pid, include_messages=True)
        if method == "GET" and route == ["messages"]:
            return to_jsonable(service.runtime.messages.list(pid, include_acked=True, limit=_query_int(query, "limit")))
        if method == "GET" and route == ["human-requests"]:
            return to_jsonable(service.runtime.human.list(pid=pid))
        if method == "GET" and route == ["llm-calls"]:
            return to_jsonable(service.runtime.store.list_llm_calls(pid=pid, limit=_query_int(query, "limit")))
        if method == "GET" and route == ["audit"]:
            return to_jsonable(
                service.runtime.audit.trace(
                    limit=_query_int(query, "limit"),
                    actor=pid,
                    target=f"process:{pid}",
                    match_any=True,
                )
            )
        if method == "GET" and route == ["events"]:
            return to_jsonable(service.runtime.events.list(target=pid))
        if method == "GET" and route == ["capabilities"]:
            return to_jsonable(service.runtime.capability.list_subject(pid, include_inactive=True))
        if method == "GET" and route == ["checkpoints"]:
            return service.runtime.checkpoint.list(pid=pid, actor=None, require_capability=False)
        if method == "POST" and route == ["run"]:
            body = self._read_body()
            return service.scheduler.start(
                pid=pid,
                max_quanta=_positive_int_or_none(body.get("max_quanta"), "max_quanta"),
                reason=f"run:{pid}",
            )
        if method == "POST" and route == ["step"]:
            return service.scheduler.run_step(pid)
        if method == "POST" and route == ["pause"]:
            body = self._read_body()
            service.runtime.process.pause(pid, str(body.get("reason") or "paused from GUI"))
            service.publish_runtime_changes("process.pause")
            return service._process_summary(pid, include_messages=True)
        if method == "POST" and route == ["resume"]:
            service.runtime.process.resume(pid)
            service.publish_runtime_changes("process.resume")
            if self._read_body(optional=True).get("auto_run", False):
                service.scheduler.maybe_start(reason=f"resume:{pid}")
            return service._process_summary(pid, include_messages=True)
        if method == "POST" and route == ["signal"]:
            body = self._read_body()
            service.runtime.process.signal(pid, ProcessSignal(str(body.get("signal") or ProcessSignal.INTERRUPT.value)), payload=body.get("payload"))
            service.publish_runtime_changes("process.signal")
            return service._process_summary(pid, include_messages=True)
        if method == "POST" and route in (["message"], ["interrupt"]):
            body = self._read_body()
            kind = ProcessMessageKind.INTERRUPT if route == ["interrupt"] else ProcessMessageKind.NORMAL
            message = service.runtime.human.send_process_message(
                pid,
                str(body.get("body") or body.get("message") or ""),
                kind=kind,
                human=str(body.get("human") or service.runtime.config.runtime.default_human),
                channel=str(body.get("channel") or "human"),
                correlation_id=body.get("correlation_id"),
                reply_to=body.get("reply_to"),
                subject=body.get("subject"),
                payload=body.get("payload") if isinstance(body.get("payload"), dict) else {},
            )
            service.publish_runtime_changes(f"process.{kind.value}_message")
            if body.get("auto_run", True):
                service.scheduler.maybe_start(
                    max_quanta=_positive_int_or_none(body.get("max_quanta"), "max_quanta"),
                    reason=f"message:{pid}",
                )
            return {"message": to_jsonable(message), "process": service._process_summary(pid, include_messages=True), "scheduler": service.scheduler.status()}
        if method == "POST" and route == ["cd"]:
            body = self._read_body()
            process = service.runtime.set_process_working_directory(pid, str(body["path"]))
            service.publish_runtime_changes("process.cd")
            return to_jsonable(process)
        if method == "POST" and route == ["exec"]:
            body = self._read_body()
            self._require_confirmed(
                "process.exec",
                body,
                {"pid": pid, "image": body.get("image"), "goal": body.get("goal"), "llm_profile": body.get("llm_profile")},
            )
            process = service.runtime.exec_process(
                pid,
                str(body["image"]),
                args=body.get("args") if isinstance(body.get("args"), dict) else {},
                goal=body.get("goal"),
                preserve_memory=bool(body.get("preserve_memory", True)),
                preserve_capabilities=bool(body.get("preserve_capabilities", False)),
                llm_profile_id=str(body["llm_profile"]) if body.get("llm_profile") is not None else None,
            )
            service.publish_runtime_changes("process.exec")
            if body.get("auto_run", True):
                service.scheduler.maybe_start(
                    max_quanta=_positive_int_or_none(body.get("max_quanta"), "max_quanta"),
                    reason=f"exec:{pid}",
                )
            return {"process": to_jsonable(process), "scheduler": service.scheduler.status()}
        if method == "POST" and route == ["exit"]:
            body = self._read_body()
            self._require_confirmed("process.exit", body, {"pid": pid, "failed": body.get("failed", False)})
            service.runtime.process.exit(pid, failed=bool(body.get("failed", False)), message=body.get("message"))
            service.publish_runtime_changes("process.exit")
            return service._process_summary(pid, include_messages=True)
        raise GuiServerError(HTTPStatus.NOT_FOUND, "unknown process endpoint")

    def _dispatch_human(self, method: str, route: list[str]) -> Any:
        service = self.server.service
        if method == "GET" and not route:
            return to_jsonable(service.runtime.human.list())
        if method == "POST" and len(route) == 2 and route[1] == "respond":
            body = self._read_body()
            current = service.runtime.human.get(route[0])
            if current.status.value != "pending":
                raise GuiServerError(
                    HTTPStatus.CONFLICT,
                    f"human request is not pending: {route[0]} status={current.status.value}",
                )
            decision = body.get("decision") if isinstance(body.get("decision"), dict) else {}
            if body.get("answer") is not None:
                decision = {**decision, "answer": str(body["answer"])}
            if bool(body.get("approved", True)):
                request = service.runtime.human.approve(route[0], {"approved": True, "source": "gui", **decision})
            else:
                request = service.runtime.human.reject(route[0], {"approved": False, "source": "gui", **decision})
            service.publish_runtime_changes("human.respond")
            if body.get("auto_run", True):
                service.scheduler.maybe_start(reason=f"human:{route[0]}")
            return {"request": to_jsonable(request), "scheduler": service.scheduler.status()}
        raise GuiServerError(HTTPStatus.NOT_FOUND, "unknown human endpoint")

    def _dispatch_checkpoints(self, method: str, route: list[str], query: dict[str, list[str]]) -> Any:
        service = self.server.service
        if method == "GET" and not route:
            return service.runtime.checkpoint.list(pid=_query_str(query, "pid"), actor=None, require_capability=False)
        if method == "POST" and route == ["create"]:
            body = self._read_body()
            checkpoint_id = service.runtime.checkpoint.create(
                str(body["pid"]),
                str(body.get("reason") or "GUI checkpoint"),
                actor=str(body.get("actor") or "gui"),
                require_capability=body.get("actor") is not None,
            )
            service.publish_runtime_changes("checkpoint.create")
            return {"checkpoint_id": checkpoint_id}
        if method == "GET" and len(route) == 1:
            return service.runtime.checkpoint.inspect(route[0], actor=None, require_capability=False)
        if method == "GET" and len(route) == 2 and route[1] == "diff":
            return service.runtime.checkpoint.diff(route[0], actor=None, require_capability=False)
        if method == "POST" and len(route) == 2 and route[1] == "restore":
            body = self._read_body()
            self._require_confirmed("checkpoint.restore", body, {"checkpoint_id": route[0]})
            result = service.runtime.checkpoint.restore(
                str(body.get("actor") or "gui"),
                route[0],
                require_capability=body.get("actor") is not None,
            )
            service.publish_runtime_changes("checkpoint.restore")
            return result
        if method == "POST" and len(route) == 2 and route[1] == "fork":
            body = self._read_body()
            self._require_confirmed("checkpoint.fork", body, {"checkpoint_id": route[0], "parent_pid": body.get("parent_pid")})
            result = service.runtime.checkpoint.fork_from_checkpoint(
                str(body.get("actor") or "gui"),
                route[0],
                parent_pid=body.get("parent_pid"),
                require_capability=body.get("actor") is not None,
            )
            service.publish_runtime_changes("checkpoint.fork")
            return result
        raise GuiServerError(HTTPStatus.NOT_FOUND, "unknown checkpoint endpoint")

    def _dispatch_skills(self, method: str, route: list[str], query: dict[str, list[str]]) -> Any:
        service = self.server.service
        if method == "GET" and not route:
            return service.runtime.skills.discover_skills(_query_str(query, "text"), require_capability=False)
        if method == "GET" and len(route) == 1:
            return service.runtime.skills.inspect_skill(route[0], require_capability=False)
        if method == "POST" and route == ["register"]:
            body = self._read_body()
            self._require_confirmed("skill.register", body, {"path": body.get("path")})
            result = service.runtime.skills.register_skill_from_path(
                str(body["path"]),
                actor=str(body.get("actor") or "gui"),
                replace=bool(body.get("replace", False)),
                require_capability=body.get("actor") is not None,
            )
            service.publish_runtime_changes("skill.register")
            return result
        if method == "POST" and len(route) == 2 and route[1] in {"activate", "unload"}:
            body = self._read_body()
            self._require_confirmed(
                f"skill.{route[1]}",
                body,
                {"pid": body.get("pid"), "skill_id": route[0], "admin_mode": body.get("actor") is None},
            )
            require_capability = body.get("actor") is not None
            actor = str(body.get("actor") or "gui")
            if route[1] == "activate":
                result = service.runtime.skills.activate_skill(
                    str(body["pid"]),
                    route[0],
                    actor=actor,
                    require_capability=require_capability,
                )
            else:
                result = service.runtime.skills.unload_skill(
                    str(body["pid"]),
                    route[0],
                    actor=actor,
                    require_capability=require_capability,
                )
            service.publish_runtime_changes(f"skill.{route[1]}")
            return result
        raise GuiServerError(HTTPStatus.NOT_FOUND, "unknown skills endpoint")

    def _dispatch_capabilities(self, method: str, route: list[str], query: dict[str, list[str]]) -> Any:
        service = self.server.service
        if method == "GET" and not route:
            subject = _query_str(query, "subject")
            return to_jsonable(service.runtime.capability.list_subject(subject, include_inactive=True) if subject else service.runtime.store.list_capabilities())
        if method == "GET" and len(route) == 1:
            return service.runtime.capability.inspect(route[0])
        if method == "POST" and route == ["grant"]:
            body = self._read_body()
            self._require_confirmed("capability.grant", body, {"subject": body.get("subject"), "resource": body.get("resource"), "rights": body.get("rights")})
            rights = _gui_capability_rights(body.get("rights"))
            actor = body.get("actor")
            if actor is None:
                cap = service.runtime.capability.grant(
                    str(body["subject"]),
                    str(body["resource"]),
                    rights,
                    issued_by="gui",
                )
            else:
                cap = service.runtime.capability.issue(
                    actor=str(actor),
                    subject=str(body["subject"]),
                    spec=CapabilitySpec(resource=str(body["resource"]), rights=set(rights)),
                    require_authority=True,
                )
            service.publish_runtime_changes("capability.grant")
            return to_jsonable(cap)
        if method == "POST" and route == ["delegate"]:
            body = self._read_body()
            self._require_confirmed("capability.delegate", body, {"parent": body.get("parent"), "child": body.get("child"), "resource": body.get("resource"), "rights": body.get("rights")})
            actor = body.get("actor")
            parent = str(body["parent"])
            if actor is not None and parent != str(actor):
                raise CapabilityDenied("GUI actor-mode delegation may only delegate from the actor process")
            cap = service.runtime.capability.delegate(
                parent,
                str(body["child"]),
                {"resource": body["resource"], "rights": _gui_capability_rights(body.get("rights"))},
                actor=str(actor or "gui"),
            )
            service.publish_runtime_changes("capability.delegate")
            return to_jsonable(cap)
        if method == "POST" and len(route) == 2 and route[1] == "revoke":
            body = self._read_body()
            self._require_confirmed("capability.revoke", body, {"capability_id": route[0], "reason": body.get("reason")})
            cap = service.runtime.capability.revoke(
                route[0],
                revoked_by=str(body.get("actor") or "gui"),
                reason=body.get("reason"),
                require_authority=body.get("actor") is not None,
            )
            service.publish_runtime_changes("capability.revoke")
            return to_jsonable(cap)
        if method == "POST" and route == ["explain"]:
            body = self._read_body()
            return service.runtime.capability.explain_decision(str(body["subject"]), str(body["resource"]), str(body["right"]))
        raise GuiServerError(HTTPStatus.NOT_FOUND, "unknown capability endpoint")

    def _dispatch_images(self, method: str, route: list[str]) -> Any:
        service = self.server.service
        if method == "GET" and not route:
            return service.runtime.image_registry.list_images()
        if method == "GET" and len(route) == 1:
            return service.runtime.image_registry.inspect(route[0])
        if method == "POST" and route == ["register"]:
            body = self._read_body()
            self._require_confirmed(
                "image.register",
                body,
                {
                    "source": body.get("source"),
                    "replace": body.get("replace", False),
                    "admin_mode": body.get("actor") is None,
                },
            )
            if "path" in body:
                raise GuiServerError(
                    HTTPStatus.BAD_REQUEST,
                    "GUI image registration accepts package files, not host file paths",
                )
            files = self._coerce_image_package_files(body.get("files"))
            result = service.runtime.image_registry.register_from_package_files(
                files,
                actor=str(body.get("actor") or "gui"),
                replace=bool(body.get("replace", False)),
                require_capability=body.get("actor") is not None,
                source=body.get("source"),
            )
            service.publish_runtime_changes("image.register")
            return {
                "image_id": result.image.image_id,
                "name": result.image.name,
                "version": result.image.version,
                "source": result.source,
                "replaced": result.replaced,
                "boot": result.image.boot,
                "default_tools": list(result.image.default_tools),
                "default_skills": list(result.image.default_skills),
                "package_sha256": result.image.metadata.get("package_sha256"),
                "package_jit_tools": result.image.metadata.get("package_jit_tools", []),
                "required_capabilities_count": len(result.image.required_capabilities),
            }
        if method == "POST" and route == ["commit"]:
            body = self._read_body()
            self._require_confirmed(
                "image.commit",
                body,
                {
                    "checkpoint_id": body.get("checkpoint_id"),
                    "image_id": body.get("image_id"),
                    "name": body.get("name"),
                    "admin_mode": body.get("actor") is None,
                },
            )
            result = service.runtime.image_registry.commit_from_checkpoint(
                actor=str(body.get("actor") or "gui"),
                checkpoint_id=str(body["checkpoint_id"]),
                image_id=str(body["image_id"]),
                name=str(body["name"]),
                version=str(body.get("version") or "v0"),
                replace=bool(body.get("replace", False)),
                metadata=dict(body.get("metadata") or {}),
                require_capability=body.get("actor") is not None,
            )
            service.publish_runtime_changes("image.commit")
            return {
                "image_id": result.image.image_id,
                "name": result.image.name,
                "version": result.image.version,
                "replaced": result.replaced,
                "boot": result.image.boot,
                "required_capabilities_count": len(result.image.required_capabilities),
            }
        raise GuiServerError(HTTPStatus.NOT_FOUND, "unknown image endpoint")

    def _coerce_image_package_files(self, value: Any) -> dict[str, bytes | str]:
        if not isinstance(value, dict) or not value:
            raise GuiServerError(HTTPStatus.BAD_REQUEST, "image registration requires non-empty package files")
        files: dict[str, bytes | str] = {}
        for path, content in value.items():
            if not isinstance(path, str) or not path:
                raise GuiServerError(HTTPStatus.BAD_REQUEST, "image package file paths must be non-empty strings")
            if isinstance(content, str):
                files[path] = content
                continue
            if isinstance(content, dict) and isinstance(content.get("base64"), str):
                try:
                    files[path] = base64.b64decode(content["base64"], validate=True)
                except Exception as exc:
                    raise GuiServerError(HTTPStatus.BAD_REQUEST, f"invalid base64 image package file: {path}") from exc
                continue
            raise GuiServerError(HTTPStatus.BAD_REQUEST, f"image package file content must be text or base64: {path}")
        return files

    def _dispatch_jsonrpc(self, method: str, route: list[str], query: dict[str, list[str]]) -> Any:
        service = self.server.service
        if method == "GET" and not route:
            return service.runtime.jsonrpc.list_endpoints(text=_query_str(query, "text"), require_capability=False)
        if method == "GET" and len(route) == 1:
            return service.runtime.jsonrpc.inspect_endpoint(route[0], require_capability=False)
        if method == "POST" and route == ["register"]:
            body = self._read_body()
            self._require_confirmed("jsonrpc.register", body, {"source": body.get("source")})
            if "path" in body:
                raise GuiServerError(
                    HTTPStatus.BAD_REQUEST,
                    "GUI JSON-RPC registration accepts manifest_text, not host file paths",
                )
            text = body.get("manifest_text")
            if not isinstance(text, str) or not text.strip():
                raise GuiServerError(HTTPStatus.BAD_REQUEST, "JSON-RPC registration requires non-empty manifest_text")
            result = service.runtime.jsonrpc.register_endpoint_from_yaml_text(
                text,
                actor=str(body.get("actor") or "gui"),
                replace=bool(body.get("replace", False)),
                require_capability=body.get("actor") is not None,
                source=body.get("source"),
            )
            service.publish_runtime_changes("jsonrpc.register")
            return result
        if method == "POST" and len(route) == 2 and route[1] == "call":
            body = self._read_body()
            self._require_confirmed("jsonrpc.call", body, {"pid": body.get("pid"), "endpoint_id": route[0], "method_id": body.get("method_id")})
            result = service.runtime.jsonrpc.call(str(body["pid"]), route[0], str(body["method_id"]), params=body.get("params"))
            service.publish_runtime_changes("jsonrpc.call")
            return to_jsonable(result)
        raise GuiServerError(HTTPStatus.NOT_FOUND, "unknown JSON-RPC endpoint")

    def _dispatch_modules(self, method: str, route: list[str]) -> Any:
        service = self.server.service
        if method == "GET" and not route:
            return service.runtime.modules.loaded_module_summaries()
        if method == "GET" and len(route) == 1:
            return service.runtime.modules.inspect_module(route[0])
        raise GuiServerError(HTTPStatus.NOT_FOUND, "unknown module endpoint")

    def _handle_sse(self, parsed: Any) -> None:
        cursor = _int_or_none(parse_qs(parsed.query).get("cursor", ["0"])[0]) or 0
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self._send_common_headers()
        self.end_headers()
        try:
            for event in self.server.service.broadcaster.replay_after(cursor):
                self._write_sse(event)
                cursor = event.seq
            while not self.server.service._closed:
                events = self.server.service.broadcaster.wait_after(cursor, timeout_s=15)
                if not events:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                for event in events:
                    self._write_sse(event)
                    cursor = event.seq
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return

    def _schedule_server_shutdown(self) -> None:
        def shutdown_after_response() -> None:
            time.sleep(self.server.service.runtime.config.gui.http_shutdown_delay_s)
            self.server.shutdown()

        threading.Thread(target=shutdown_after_response, name="agent-libos-gui-http-shutdown", daemon=True).start()

    def _write_sse(self, event: GuiEvent) -> None:
        payload_data = _sse_payload_data(
            event.data,
            max_bytes=self.server.service.runtime.config.gui.sse_payload_max_bytes,
            string_limit=self.server.service.runtime.config.gui.snapshot_string_max_chars,
            collection_limit=self.server.service.runtime.config.gui.snapshot_collection_max_items,
        )
        payload = json.dumps(payload_data, ensure_ascii=False, default=str)
        self.wfile.write(f"id: {event.seq}\nevent: {event.event}\ndata: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _read_body(self, optional: bool = False) -> dict[str, Any]:
        if getattr(self, "_body_cached", False):
            return dict(getattr(self, "_cached_json_body", {}))
        raw_length = self.headers.get("Content-Length", "0") or "0"
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise GuiServerError(HTTPStatus.BAD_REQUEST, "invalid Content-Length header") from exc
        if length < 0:
            raise GuiServerError(HTTPStatus.BAD_REQUEST, "invalid Content-Length header")
        request_body_max_bytes = self.server.service.runtime.config.gui.request_body_max_bytes
        if length > request_body_max_bytes:
            # Drain small rejected bodies so clients get the 413 JSON response
            # instead of a TCP reset. Very large bodies are still closed early
            # to keep the GUI facade from becoming an unbounded discard sink.
            reject_drain_limit = max(request_body_max_bytes * 2, 64 * 1024)
            if length <= reject_drain_limit:
                self.rfile.read(length)
            else:
                self.close_connection = True
            raise GuiServerError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"request body exceeds {request_body_max_bytes} bytes",
            )
        if length == 0:
            return {} if optional else {}
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise GuiServerError(HTTPStatus.BAD_REQUEST, f"invalid JSON body: {exc}") from exc
        if not isinstance(value, dict):
            raise GuiServerError(HTTPStatus.BAD_REQUEST, "request body must be a JSON object")
        return value

    def _require_auth(self) -> None:
        token = self.server.service.token
        header = self.headers.get("Authorization", "")
        if not secrets.compare_digest(header, f"Bearer {token}"):
            raise GuiServerError(HTTPStatus.UNAUTHORIZED, "missing or invalid GUI session token")

    def _require_confirmed(self, action: str, body: dict[str, Any], preview: dict[str, Any]) -> None:
        if body.get("confirmed") is True:
            return
        self.server.service.runtime.audit.record(
            actor="gui",
            action="gui.confirmation_required",
            target=action,
            decision={"preview": preview},
        )
        raise GuiServerError(
            HTTPStatus.CONFLICT,
            f"{action} requires explicit confirmation",
            details={"confirmation_required": True, "action": action, "preview": preview},
        )

    def _write_json(self, value: Any, *, status: int = HTTPStatus.OK) -> None:
        payload = json.dumps(to_jsonable(value), ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self._send_common_headers()
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    def _send_common_headers(self) -> None:
        origin = _allowed_cors_origin(self.headers.get("Origin"))
        if origin is not None:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        if self.close_connection:
            self.send_header("Connection", "close")


def create_gui_http_server(
    *,
    db: str = _RUNTIME_DEFAULTS.local_store_target,
    host: str = "127.0.0.1",
    port: int = 0,
    token: str | None = None,
    auto_run: bool = True,
    max_quanta: int | None | object = _CONFIG_DEFAULT,
    runtime: Runtime | None = None,
) -> GuiHTTPServer:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("GUI server is local-only and must bind 127.0.0.1")
    service = GuiRuntimeService(db=db, runtime=runtime, token=token, auto_run=auto_run, max_quanta=max_quanta)
    return GuiHTTPServer(("127.0.0.1", int(port)), service)


def serve(
    *,
    db: str,
    port: int,
    token: str | None,
    auto_run: bool,
    max_quanta: int | None | object,
    ready: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    server = create_gui_http_server(db=db, port=port, token=token, auto_run=auto_run, max_quanta=max_quanta)
    host, selected_port = server.server_address
    payload = {"url": f"http://{host}:{selected_port}", "token": server.service.token, "db": db}
    if ready is not None:
        ready(payload)
    else:
        print(json.dumps(payload, ensure_ascii=True), flush=True)
    try:
        server.serve_forever()
    finally:
        server.service.shutdown()
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agent-libos-gui-server")
    parser.add_argument("--db", default=_RUNTIME_DEFAULTS.local_store_target)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--token")
    parser.add_argument("--no-auto-run", action="store_true")
    parser.add_argument(
        "--max-quanta",
        type=int,
        default=_CONFIG_DEFAULT,
        help="Optional default quantum budget for GUI scheduler runs; omitted uses runtime config.",
    )
    args = parser.parse_args(argv)
    if args.max_quanta is not _CONFIG_DEFAULT and args.max_quanta <= 0:
        parser.error("--max-quanta must be a positive integer when provided")
    serve(
        db=args.db,
        port=args.port,
        token=args.token,
        auto_run=not args.no_auto_run,
        max_quanta=args.max_quanta,
    )


def _int_or_none(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise GuiServerError(HTTPStatus.BAD_REQUEST, "integer value expected") from exc


def _positive_int_or_none(value: Any, name: str) -> int | None:
    parsed = _int_or_none(value)
    if parsed is None:
        return None
    if parsed <= 0:
        raise GuiServerError(HTTPStatus.BAD_REQUEST, f"{name} must be a positive integer or omitted")
    return parsed


def _gui_capability_rights(value: Any) -> list[str]:
    if value is None:
        return [CapabilityRight.READ.value]
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _bounded_float_or_default(value: Any, name: str, *, default: float, maximum: float) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise GuiServerError(HTTPStatus.BAD_REQUEST, f"{name} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise GuiServerError(HTTPStatus.BAD_REQUEST, f"{name} must be a finite number")
    if parsed < 0:
        raise GuiServerError(HTTPStatus.BAD_REQUEST, f"{name} must be non-negative")
    if parsed > maximum:
        raise GuiServerError(HTTPStatus.BAD_REQUEST, f"{name} must be at most {maximum} seconds")
    return parsed


def _query_int(query: dict[str, list[str]], key: str) -> int | None:
    values = query.get(key)
    return _int_or_none(values[0]) if values else None


def _query_str(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values else None


def _is_object_task_wait_request(method: str, path: str) -> bool:
    if method != "POST":
        return False
    parts = [unquote(part) for part in path.strip("/").split("/") if part]
    return len(parts) == 4 and parts[:2] == ["api", "object-tasks"] and parts[3] == "wait"


def _object_task_owner_handle(
    runtime: Runtime,
    pid: str,
    owner_oid: Any,
    owner_name: Any,
    namespace: Any,
):
    if owner_oid is not None:
        return runtime.memory.handle_for_oid(
            pid,
            str(owner_oid),
            required_rights={ObjectRight.READ.value, ObjectRight.WRITE.value, ObjectRight.LINK.value},
        )
    if owner_name is not None:
        return runtime.memory.handle_for_name(
            pid,
            str(owner_name),
            rights={ObjectRight.READ.value, ObjectRight.WRITE.value, ObjectRight.LINK.value},
            namespace=str(namespace) if namespace is not None else None,
        )
    raise GuiServerError(HTTPStatus.BAD_REQUEST, "owner_oid or owner_name is required")


def _object_task_owner_watch_body(body: dict[str, Any]) -> dict[str, Any] | bool:
    raw_events = body.get("watch_events")
    if raw_events is not None and not isinstance(raw_events, list):
        raise GuiServerError(HTTPStatus.BAD_REQUEST, "watch_events must be a JSON array")
    events = [str(item) for item in raw_events] if raw_events is not None else []
    enabled = bool(body.get("owner_watch", False) or events or body.get("watch_channel") or "watch_kind" in body)
    if not enabled:
        return False
    selected: dict[str, Any] = {
        "enabled": True,
        "kind": str(body.get("watch_kind") or ProcessMessageKind.NORMAL.value),
    }
    if events:
        selected["events"] = events
    if body.get("watch_channel") is not None:
        selected["channel"] = str(body["watch_channel"])
    return selected


def _allowed_cors_origin(origin: str | None) -> str | None:
    if not origin:
        return None
    parsed = urlparse(origin)
    if parsed.scheme in {"http", "https"} and parsed.hostname in {"127.0.0.1", "localhost"}:
        return origin
    return None
