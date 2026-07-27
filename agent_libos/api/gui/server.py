from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import secrets
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qs, unquote, urlparse

from pydantic import ValidationError as PydanticValidationError

from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig, load_config_file, load_config_from_project_root
from agent_libos.llm.user_profiles import (
    UserLLMProfileStore,
    default_user_llm_profiles_path,
    normalize_user_llm_profile_id,
    serialize_user_llm_profile,
    summarize_llm_profile,
)
from agent_libos.models import (
    CapabilityRight,
    CapabilitySpec,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    ObjectRight,
    ProcessMessageKind,
    ProcessSignal,
    ProcessStatus,
    process_state_to_mapping,
)
from agent_libos.storage.gui_visibility import (
    is_gui_presentation_audit,
    is_gui_presentation_event,
)
from agent_libos.models.exceptions import (
    CapabilityDenied,
    HumanApprovalRequired,
    NotFound,
    ProcessMessageWaitRequired,
    ProcessWaitRequired,
    ValidationError,
)
from agent_libos.runtime.runtime import Runtime
from agent_libos.storage import display_store_target
from agent_libos.utils.public_errors import (
    internal_exception_observation,
    public_error_envelope,
)
from agent_libos.utils.serde import bounded_json_loads, to_jsonable

_GUI_DEFAULTS = DEFAULT_CONFIG.gui
_GUI_PRODUCTION_RENDERER_ORIGIN = "agent-libos://app"
_TERMINAL = {ProcessStatus.EXITED, ProcessStatus.FAILED, ProcessStatus.KILLED}
_CONFIG_DEFAULT = object()
_SUMMARY_UNSET = object()
_GUI_BOOL_FIELDS = {
    "approved",
    "allow_custom_base_url",
    "auto_run",
    "auto_wait_on_empty_tool_calls",
    "fallback_json_actions",
    "confirmed",
    "enabled",
    "failed",
    "grant_result_to_notify",
    "owner_watch",
    "parallel_tool_calls",
    "preserve_capabilities",
    "preserve_memory",
    "replace",
    "responses_previous_response_id",
    "store",
}
_GUI_NULLABLE_BOOL_FIELDS = {
    "allow_custom_base_url",
    "auto_wait_on_empty_tool_calls",
    "fallback_json_actions",
    "parallel_tool_calls",
    "responses_previous_response_id",
    "store",
}


class _GuiHumanPresentationProvider:
    """The protected handoff from Host-owned state to GUI response JSON."""

    @staticmethod
    def present(view: dict[str, Any]) -> dict[str, Any]:
        return view

    @staticmethod
    def classify_external_effect(
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification:
        if operation != "write":
            raise ValueError(f"unsupported GUI Human presentation operation: {operation}")
        return ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
            rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
            state_mutation=False,
            information_flow=True,
            metadata={
                "channel": "gui",
                "request_id": context.get("request_id"),
                "presented": isinstance(result, dict),
            },
        )


def _load_runtime_config(config_path: str | None, parser: argparse.ArgumentParser) -> AgentLibOSConfig:
    try:
        if config_path:
            return load_config_file(config_path)
        return load_config_from_project_root()
    except (OSError, ValueError, PydanticValidationError) as exc:
        parser.error(str(exc))
    raise AssertionError("argparse parser.error should exit")


def _bounded_gui_value(
    value: Any,
    *,
    string_limit: int,
    collection_limit: int,
    truncated: dict[str, Any] | None = None,
    path: str = "$",
) -> Any:
    jsonable = to_jsonable(value)
    if isinstance(jsonable, str):
        if len(jsonable) <= string_limit:
            return jsonable
        if truncated is not None:
            truncated[path] = {
                "kind": "string",
                "returned": string_limit,
                "chars": len(jsonable),
                "omitted": len(jsonable) - string_limit,
            }
        return jsonable[:string_limit]
    if isinstance(jsonable, list):
        selected = [
            _bounded_gui_value(
                item,
                string_limit=string_limit,
                collection_limit=collection_limit,
                truncated=truncated,
                path=f"{path}[{index}]",
            )
            for index, item in enumerate(jsonable[:collection_limit])
        ]
        if len(jsonable) > collection_limit:
            if truncated is not None:
                truncated[path] = {
                    "kind": "array",
                    "returned": collection_limit,
                    "omitted": len(jsonable) - collection_limit,
                }
        return selected
    if isinstance(jsonable, dict):
        items = list(jsonable.items())
        bounded = {
            str(key): _bounded_gui_value(
                item,
                string_limit=string_limit,
                collection_limit=collection_limit,
                truncated=truncated,
                path=f"{path}.{key}" if path != "$" else str(key),
            )
            for key, item in items[:collection_limit]
        }
        if len(items) > collection_limit:
            if truncated is not None:
                truncated[path] = {
                    "kind": "object",
                    "returned": collection_limit,
                    "omitted": len(items) - collection_limit,
                }
        return bounded
    return jsonable


def _bounded_gui_payload(
    value: Any,
    *,
    string_limit: int,
    collection_limit: int,
    pre_truncated: dict[str, Any] | None = None,
) -> Any:
    truncated: dict[str, Any] = {}
    bounded = _bounded_gui_value(
        value,
        string_limit=string_limit,
        collection_limit=collection_limit,
        truncated=truncated,
    )
    if isinstance(bounded, dict):
        combined = {**truncated, **dict(pre_truncated or {})}
        if combined:
            bounded["_truncated"] = combined
    return bounded


def _take_source_window(
    values: list[Any],
    *,
    limit: int,
    path: str,
    truncated: dict[str, Any],
    source_has_more: bool = False,
) -> list[Any]:
    """Clip a bounded source window and preserve its next-row signal."""

    if len(values) <= limit and not source_has_more:
        return values
    truncated[path] = {
        "kind": "array",
        "returned": min(len(values), limit),
        "omitted": max(0, len(values) - limit) + (1 if source_has_more else 0),
        "omitted_is_lower_bound": True,
        "source_limited": True,
    }
    return values[:limit]


def _sse_payload_data(
    event: str,
    data: dict[str, Any],
    *,
    max_bytes: int,
    string_limit: int,
    collection_limit: int,
) -> tuple[str, dict[str, Any]]:
    bounded = _bounded_gui_payload(data, string_limit=string_limit, collection_limit=collection_limit)
    encoded = json.dumps(bounded, ensure_ascii=False, default=str).encode("utf-8")
    if len(encoded) <= max_bytes:
        return event, bounded
    invalidation_event = "snapshot_truncated" if event == "snapshot" else "event.invalidated"
    return invalidation_event, {
        "invalidated": True,
        "event": event,
        "bytes": len(encoded),
        "reason": "gui event payload exceeds sse_payload_max_bytes",
    }


def _json_bool(body: dict[str, Any], key: str, default: bool) -> bool:
    if key not in body:
        return default
    value = body[key]
    if not isinstance(value, bool):
        raise GuiServerError(HTTPStatus.BAD_REQUEST, f"{key} must be a JSON boolean")
    return value


def _validate_json_bool_fields(body: dict[str, Any]) -> None:
    for key in sorted(_GUI_BOOL_FIELDS.intersection(body)):
        if body[key] is None and key in _GUI_NULLABLE_BOOL_FIELDS:
            continue
        if not isinstance(body[key], bool):
            raise GuiServerError(HTTPStatus.BAD_REQUEST, f"{key} must be a JSON boolean")


def _required_body_string(body: dict[str, Any], key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value.strip():
        raise GuiServerError(HTTPStatus.BAD_REQUEST, f"{key} must be a non-empty JSON string")
    return value


def _required_package_sha256(body: dict[str, Any], key: str) -> str:
    value = _required_body_string(body, key)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise GuiServerError(
            HTTPStatus.BAD_REQUEST,
            f"{key} must be a lowercase SHA-256 hex digest",
        )
    return value


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
        if max_events <= 0:
            raise ValueError("GUI event buffer limit must be positive")
        self._condition = threading.Condition()
        self._events: deque[GuiEvent] = deque(maxlen=max_events)
        self._next_seq = 1
        self._closed = False

    def publish(self, event: str, data: dict[str, Any] | None = None) -> GuiEvent:
        with self._condition:
            item = GuiEvent(seq=self._next_seq, event=event, data=data or {})
            self._next_seq += 1
            self._events.append(item)
            self._condition.notify_all()
            return item

    def replay_after(self, cursor: int) -> list[GuiEvent]:
        with self._condition:
            return self._events_after_locked(cursor)

    def wait_after(self, cursor: int, timeout_s: float = 15.0) -> list[GuiEvent]:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while not self._closed:
                ready = self._events_after_locked(cursor)
                if ready:
                    return ready
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self._condition.wait(remaining)
            return []

    def _events_after_locked(self, cursor: int) -> list[GuiEvent]:
        """Return replayable events and make an evicted/restarted cursor explicit.

        Sequence numbers are intentionally in-memory and restart at one with a
        new GUI server.  A client can therefore present either a cursor older
        than the retained buffer or one ahead of this server's newest event.
        Silently returning the retained suffix in the first case (or nothing in
        the second) can leave the renderer permanently stale.  The synthetic
        invalidation tells it to fetch a fresh snapshot before applying the
        retained stream.
        """

        if not self._events:
            if cursor <= 0:
                return []
            return [self._cursor_invalidation(cursor, reset_cursor=0, oldest=None, latest=None)]

        oldest = self._events[0].seq
        latest = self._events[-1].seq
        if cursor < oldest - 1:
            reset_cursor = oldest - 1
            return [
                self._cursor_invalidation(cursor, reset_cursor=reset_cursor, oldest=oldest, latest=latest),
                *self._events,
            ]
        if cursor > latest:
            return [
                self._cursor_invalidation(cursor, reset_cursor=0, oldest=oldest, latest=latest),
                *self._events,
            ]
        return [event for event in self._events if event.seq > cursor]

    @staticmethod
    def _cursor_invalidation(
        requested_cursor: int,
        *,
        reset_cursor: int,
        oldest: int | None,
        latest: int | None,
    ) -> GuiEvent:
        return GuiEvent(
            seq=reset_cursor,
            event="event.invalidated",
            data={
                "invalidated": True,
                "reason": "sse_cursor_not_replayable",
                "requested_cursor": requested_cursor,
                "reset_cursor": reset_cursor,
                "oldest_available": oldest,
                "latest_available": latest,
            },
        )

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class _BoundedSeenKeys:
    """A bounded insertion-ordered set for GUI delta de-duplication."""

    def __init__(self, limit: int) -> None:
        self._limit = max(1, limit)
        self._keys: OrderedDict[str, None] = OrderedDict()

    def add_if_new(self, key: str) -> bool:
        if key in self._keys:
            self._keys.move_to_end(key)
            return False
        self._keys[key] = None
        while len(self._keys) > self._limit:
            self._keys.popitem(last=False)
        return True

    def __len__(self) -> int:
        return len(self._keys)


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
            self.auto_run = enabled
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
            self.paused = False
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
            # Capture the start acknowledgement while the worker is still
            # excluded by this lock. A fast terminal batch may finish while
            # status publication is in progress, but that must not rewrite the
            # response to the request that successfully started it.
            started_status = self.status()
        self.service.publish_scheduler_status()
        return started_status

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
                with self._lock:
                    self.last_result = [result]
        except Exception as exc:
            public_error = public_error_envelope(exc)
            with self._lock:
                self.last_error = public_error["message"]
            raise
        finally:
            with self._lock:
                self.running = False
                self.finished_at = time.time()
            self.service.publish_scheduler_status()
        # Publish the snapshot only after the synchronous step has transitioned
        # back to its final scheduler state.  Otherwise both the API response
        # and the renderer's latest snapshot can claim that a completed step is
        # still running.
        self.service.publish_runtime_changes("step")
        return {"started": True, "result": to_jsonable(result), "scheduler": self.status()}

    def _run_background(self, max_quanta: int | None, pid: str | None) -> None:
        collected: list[Any] = []
        remaining = max_quanta
        try:
            while True:
                with self._lock:
                    if self.paused:
                        break
                if remaining is not None and remaining <= 0:
                    break
                batch_quanta = 1 if remaining is None else min(1, remaining)
                with self.service.runtime_user():
                    result = (
                        self.service.runtime.run_process_until_idle(pid, max_quanta=batch_quanta)
                        if pid is not None
                        else self.service.runtime.run_until_idle(
                            max_quanta=batch_quanta,
                            process_human_queue=False,
                            cancel_inflight_on_budget_exhaustion=False,
                        )
                    )
                if not result:
                    break
                collected.extend(result)
                with self._lock:
                    self.last_result = list(collected)
                self.service.publish_runtime_changes("scheduler.batch")
                if remaining is not None:
                    remaining -= len(result)
        except Exception as exc:  # pragma: no cover - covered through API status assertions
            public_error = self.service.record_internal_error(
                exc,
                action="gui.scheduler_background_internal_error",
                target="scheduler",
            )
            with self._lock:
                self.last_error = public_error["message"]
        finally:
            with self._lock:
                self.last_result = list(collected)
                self.running = False
                self.finished_at = time.time()
            self.service.publish_runtime_changes("scheduler")
            self.service.publish_scheduler_status()


class GuiRuntimeService:
    """Local-only HTTP facade over one Agent libOS Runtime instance."""

    def __init__(
        self,
        *,
        db: str | None = None,
        runtime: Runtime | None = None,
        config: AgentLibOSConfig | None = None,
        token: str | None = None,
        auto_run: bool = True,
        max_quanta: int | None | object = _CONFIG_DEFAULT,
        llm_profiles_file: str | Path | None = None,
    ) -> None:
        if runtime is not None:
            if config is not None and config != runtime.config:
                raise ValidationError(
                    "explicit GUI config must match the supplied Runtime config"
                )
            selected_config = runtime.config
        else:
            selected_config = config or DEFAULT_CONFIG
        user_llm_profiles = UserLLMProfileStore(
            llm_profiles_file,
            config=selected_config,
        )
        loaded_user_llm_profiles = user_llm_profiles.load()
        conflicts = sorted(
            set(loaded_user_llm_profiles) & set(selected_config.llm.profiles)
        )
        if conflicts:
            raise ValidationError(
                "user LLM profiles cannot override config profiles: "
                + ", ".join(conflicts)
            )
        self._db_target = db
        if runtime is None:
            self.db = display_store_target(db, config=selected_config)
            self.runtime = Runtime.open(db, config=selected_config)
        else:
            display_target = db if db is not None else runtime.store.path
            self.db = display_store_target(display_target, config=selected_config)
            self.runtime = runtime
        self.audit = self.runtime.audit
        self.owns_runtime = runtime is None
        try:
            self._initialize_service_state(
                token=token,
                auto_run=auto_run,
                max_quanta=max_quanta,
                user_llm_profiles=user_llm_profiles,
                loaded_user_llm_profiles=loaded_user_llm_profiles,
            )
        except BaseException:
            self._cleanup_failed_initialization()
            raise

    def _initialize_service_state(
        self,
        *,
        token: str | None,
        auto_run: bool,
        max_quanta: int | None | object,
        user_llm_profiles: UserLLMProfileStore,
        loaded_user_llm_profiles: dict[str, Any],
    ) -> None:
        self.token = token or secrets.token_urlsafe(32)
        self._human_presentation_provider = _GuiHumanPresentationProvider()
        self.broadcaster = GuiEventBroadcaster(max_events=self.runtime.config.gui.event_buffer_limit)
        self.runtime_lock = threading.RLock()
        self._lifecycle = threading.Condition(threading.RLock())
        self._active_runtime_users = 0
        self._closing = False
        self._shutdown_in_progress = False
        self._runtime_teardown_started = False
        selected_max_quanta = self.runtime.config.runtime.run_until_idle_max_quanta if max_quanta is _CONFIG_DEFAULT else max_quanta
        self.scheduler = SchedulerController(self, auto_run=auto_run, default_max_quanta=selected_max_quanta, paused=not auto_run)
        self._closed = False
        self._static_snapshot_cache: dict[str, Any] | None = None
        self._static_snapshot_truncated: dict[str, Any] = {}
        self._static_snapshot_dirty = True
        dedupe_limit = max(1, self.runtime.config.gui.event_buffer_limit * 2)
        self._seen_event_ids = _BoundedSeenKeys(max(dedupe_limit, self.runtime.config.gui.snapshot_event_limit * 2))
        self._seen_audit_ids = _BoundedSeenKeys(max(dedupe_limit, self.runtime.config.gui.snapshot_audit_limit * 2))
        self._seen_human_request_versions = _BoundedSeenKeys(
            max(dedupe_limit, self.runtime.config.gui.snapshot_collection_max_items * 2)
        )
        self._seen_message_ids = _BoundedSeenKeys(
            max(
                dedupe_limit,
                self.runtime.config.gui.snapshot_collection_max_items
                * self.runtime.config.gui.snapshot_process_message_limit,
            )
        )
        self._seen_llm_call_ids = _BoundedSeenKeys(
            max(dedupe_limit, self.runtime.config.gui.snapshot_llm_call_limit * 2)
        )
        self.user_llm_profiles = user_llm_profiles
        self._user_llm_profile_cache = self._register_user_llm_profiles(
            loaded_user_llm_profiles
        )
        self.publish_runtime_changes("startup")

    def _cleanup_failed_initialization(self) -> None:
        broadcaster = getattr(self, "broadcaster", None)
        if broadcaster is not None:
            broadcaster.close()
        if not self.owns_runtime:
            return
        try:
            self.runtime.shutdown(
                actor="gui-server",
                reason="gui-server.initialization_failed",
            )
        except BaseException:
            pass

    @contextmanager
    def runtime_user(self, *, serialize: bool = True) -> Iterator[None]:
        """Register an in-flight Runtime user so shutdown cannot close under it."""
        with self._lifecycle:
            if self._closed or self._closing:
                raise GuiServerError(HTTPStatus.SERVICE_UNAVAILABLE, "GUI runtime is shutting down")
            self._active_runtime_users += 1
        try:
            if serialize:
                with self.runtime_lock:
                    yield
            else:
                yield
        finally:
            with self._lifecycle:
                self._active_runtime_users -= 1
                self._lifecycle.notify_all()

    def shutdown(self, timeout_s: float | None = None) -> bool:
        selected_timeout = (
            self.runtime.config.gui.scheduler_shutdown_join_timeout_s
            if timeout_s is None
            else max(0.0, float(timeout_s))
        )
        deadline = time.monotonic() + selected_timeout
        with self._lifecycle:
            if self._closed:
                return True
            if self._shutdown_in_progress:
                return False
            self._closing = True
            self._shutdown_in_progress = True

        completed = False
        try:
            scheduler_timeout = max(0.0, deadline - time.monotonic())
            if not self.scheduler.shutdown(timeout_s=scheduler_timeout):
                return False
            with self._lifecycle:
                while self._active_runtime_users:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._lifecycle.wait(timeout=remaining)
            # Exclude untracked legacy callers that still use runtime_lock
            # directly while the owned Runtime releases its store handles.
            remaining = max(0.0, deadline - time.monotonic())
            acquired = self.runtime_lock.acquire(timeout=remaining)
            if not acquired:
                return False
            try:
                if self.owns_runtime:
                    # Runtime.shutdown is a phased, stateful teardown.  Once it
                    # starts, a false/exception result must not reopen the HTTP
                    # API onto a partially stopped Runtime.  A later shutdown
                    # call may retry teardown, but runtime_user stays closed.
                    self._runtime_teardown_started = True
                    result = self.runtime.shutdown(actor="gui-server", reason="gui-server.shutdown")
                    if result.get("ok") is not True:
                        return False
            finally:
                self.runtime_lock.release()
            self.broadcaster.close()
            with self._lifecycle:
                self._closed = True
            completed = True
            return True
        finally:
            with self._lifecycle:
                self._shutdown_in_progress = False
                if not completed and not self._runtime_teardown_started:
                    self._closing = False
                self._lifecycle.notify_all()

    def close(self) -> None:
        self.shutdown()

    @property
    def closed(self) -> bool:
        with self._lifecycle:
            return self._closed

    def publish_scheduler_status(self) -> None:
        self.broadcaster.publish("scheduler.status", self.scheduler.status())

    def publish_runtime_changes(self, reason: str) -> None:
        with self.runtime_lock:
            if self._reason_changes_static_snapshot(reason):
                self._static_snapshot_dirty = True
            snapshot = self.snapshot()
            self.broadcaster.publish("snapshot", {"reason": reason, "snapshot": snapshot})
            for event in snapshot["events"]:
                if not self._seen_event_ids.add_if_new(event["event_id"]):
                    continue
                self.broadcaster.publish("event.appended", event)
            for record in snapshot["audit"]:
                if not self._seen_audit_ids.add_if_new(record["record_id"]):
                    continue
                self.broadcaster.publish("audit.appended", record)
            for request in snapshot["human_requests"]:
                version_key = ":".join(
                    (
                        str(request["request_id"]),
                        str(request.get("updated_at") or ""),
                        str(request.get("status") or ""),
                    )
                )
                if not self._seen_human_request_versions.add_if_new(version_key):
                    continue
                self.broadcaster.publish("human_request.updated", request)
            for process in snapshot["processes"]:
                for message in process.get("messages", []):
                    if not self._seen_message_ids.add_if_new(message["message_id"]):
                        continue
                    self.broadcaster.publish("message.posted", message)
            for call in snapshot["llm_calls"]:
                if not self._seen_llm_call_ids.add_if_new(call["call_id"]):
                    continue
                self.broadcaster.publish("llm_call.appended", call)

    def health(self) -> dict[str, Any]:
        process_count: int | None = None
        runtime_busy = not self.runtime_lock.acquire(blocking=False)
        if not runtime_busy:
            try:
                process_count = len(self.runtime.process.list())
            finally:
                self.runtime_lock.release()
        return {
            "ok": True,
            "db": self.db,
            "scheduler": self.scheduler.status(),
            "process_count": process_count,
            "runtime_busy": runtime_busy,
        }

    def record_internal_error(
        self,
        error: BaseException,
        *,
        action: str,
        target: str,
    ) -> dict[str, str]:
        public_error = public_error_envelope(error)
        try:
            self.audit.record(
                actor="gui",
                action=action,
                target=target,
                decision={
                    "public_error": dict(public_error),
                    "internal_error": internal_exception_observation(
                        error,
                        correlation_id=public_error["correlation_id"],
                    ),
                },
                correlation_id=public_error["correlation_id"],
            )
        except Exception:
            # The outward boundary must remain safe even when the failing
            # dependency is the audit store itself.
            pass
        return public_error

    def snapshot(self) -> dict[str, Any]:
        with self.runtime_lock:
            collection_limit = self.runtime.config.gui.snapshot_collection_max_items
            source_truncated: dict[str, Any] = {}
            processes = self._process_summaries(
                limit=collection_limit,
                include_messages=True,
                truncated=source_truncated,
            )
            projected_human_requests, human_requests_have_more = (
                self.runtime.human.list_for_presentation_window(
                    presentation="gui",
                    provider=self._human_presentation_provider,
                    limit=collection_limit,
                )
            )
            human_requests = _take_source_window(
                projected_human_requests,
                limit=collection_limit,
                path="human_requests",
                truncated=source_truncated,
                source_has_more=human_requests_have_more,
            )
            static = self._static_snapshot()
            source_truncated.update(self._static_snapshot_truncated)
            snapshot = {
                "db": self.db,
                "scheduler": self.scheduler.status(),
                "processes": processes,
                "human_requests": human_requests,
                "events": to_jsonable(self._snapshot_events()),
                "audit": to_jsonable(self._snapshot_audit()),
                "llm_calls": to_jsonable(self.runtime.store.list_llm_calls(limit=self.runtime.config.gui.snapshot_llm_call_limit)),
                "object_tasks": to_jsonable(self.runtime.object_tasks.list(limit=self.runtime.config.gui.snapshot_object_task_limit)),
                **static,
            }
            return self._bounded_snapshot(snapshot, source_truncated=source_truncated)

    def _snapshot_events(self) -> list[Any]:
        limit = self.runtime.config.gui.snapshot_event_limit
        return self.runtime.events.list(
            limit=limit,
            include_gui_presentation=False,
        )

    def _snapshot_audit(self) -> list[Any]:
        limit = self.runtime.config.gui.snapshot_audit_limit
        return self.audit.trace(
            limit=limit,
            include_gui_presentation=False,
        )

    @staticmethod
    def _is_gui_presentation_event(event: Any) -> bool:
        return is_gui_presentation_event(event)

    @staticmethod
    def _is_gui_presentation_audit(record: Any) -> bool:
        return is_gui_presentation_audit(record)

    def human_request_views(
        self,
        *,
        pid: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.runtime.human.list_for_presentation(
            presentation="gui",
            provider=self._human_presentation_provider,
            pid=pid,
            limit=limit,
        )

    def human_request_view(self, request: Any) -> dict[str, Any]:
        return self.runtime.human.present_request_view(
            request,
            presentation="gui",
            provider=self._human_presentation_provider,
        )

    def _static_snapshot(self) -> dict[str, Any]:
        if self._static_snapshot_cache is None or self._static_snapshot_dirty:
            limit = self.runtime.config.gui.snapshot_collection_max_items
            fetch_limit = limit + 1
            truncated: dict[str, Any] = {}
            skills, skills_have_more = self.runtime.skills.discover_skills_window(
                require_capability=False,
                limit=min(fetch_limit, self.runtime.config.skills.discover_limit),
            )
            jsonrpc_endpoints, jsonrpc_endpoints_have_more = self.runtime.jsonrpc.list_endpoints_window(
                require_capability=False,
                limit=min(fetch_limit, self.runtime.config.jsonrpc.list_limit),
            )
            mcp_servers, mcp_servers_have_more = self.runtime.mcp.list_servers_window(
                require_capability=False,
                limit=min(fetch_limit, self.runtime.config.mcp.list_limit),
            )
            self._static_snapshot_cache = {
                "tools": _take_source_window(
                    self._tool_summaries(limit=fetch_limit),
                    limit=limit,
                    path="tools",
                    truncated=truncated,
                ),
                "images": _take_source_window(
                    to_jsonable(self.runtime.image_registry.list_images(limit=fetch_limit)),
                    limit=limit,
                    path="images",
                    truncated=truncated,
                ),
                "skills": _take_source_window(
                    to_jsonable(skills),
                    limit=limit,
                    path="skills",
                    truncated=truncated,
                    source_has_more=skills_have_more,
                ),
                "jsonrpc_endpoints": _take_source_window(
                    to_jsonable(jsonrpc_endpoints),
                    limit=limit,
                    path="jsonrpc_endpoints",
                    truncated=truncated,
                    source_has_more=jsonrpc_endpoints_have_more,
                ),
                "mcp_servers": _take_source_window(
                    to_jsonable(mcp_servers),
                    limit=limit,
                    path="mcp_servers",
                    truncated=truncated,
                    source_has_more=mcp_servers_have_more,
                ),
                "modules": _take_source_window(
                    to_jsonable(self.runtime.modules.loaded_module_summaries(limit=fetch_limit)),
                    limit=limit,
                    path="modules",
                    truncated=truncated,
                ),
                "llm_profiles": _take_source_window(
                    self._llm_profile_summaries(limit=fetch_limit),
                    limit=limit,
                    path="llm_profiles",
                    truncated=truncated,
                ),
            }
            self._static_snapshot_truncated = truncated
            self._static_snapshot_dirty = False
        return dict(self._static_snapshot_cache)

    def _reason_changes_static_snapshot(self, reason: str) -> bool:
        return reason.startswith(("image.", "skill.", "jsonrpc.", "mcp.", "module.", "process.exec", "llm_profile."))

    def _process_summary(
        self,
        pid: str,
        *,
        include_messages: bool = False,
        process: Any | None = None,
        activity: dict[str, Any] | None = None,
        resource_remaining: Any | None = None,
        rating: Any = _SUMMARY_UNSET,
    ) -> dict[str, Any]:
        process = process if process is not None else self.runtime.process.get(pid)
        selected_activity = (
            activity
            if activity is not None
            else self.runtime.store.get_process_activity_summaries(
                [pid],
                recent_message_limit=(
                    self.runtime.config.gui.snapshot_process_message_limit if include_messages else 0
                ),
                recent_llm_call_limit=self.runtime.config.gui.snapshot_process_llm_call_limit,
            )
        )
        activity_row = selected_activity.get(
            pid,
            {
                "unread_message_count": 0,
                "interrupt_count": 0,
                "llm_call_count": 0,
                "token_total": 0,
                "messages": [],
            },
        )
        # Activity rows intentionally read only a bounded recent LLM window.
        # Use the process' durable, hierarchical resource counters for the
        # user-facing totals so long-running tasks are not reported as if they
        # stopped at ``snapshot_process_llm_call_limit``.  The bounded values
        # remain a compatibility floor for imported/manual call rows that
        # predate resource accounting.
        llm_call_count = max(
            int(process.resource_usage.llm_calls),
            int(activity_row["llm_call_count"]),
        )
        token_total = max(
            int(process.resource_usage.llm_total_tokens),
            int(activity_row["token_total"]),
        )
        return {
            **to_jsonable(process),
            **process_state_to_mapping(
                process.status.value,
                process.wait_state,
                process.outcome,
                process.state_generation,
            ),
            "terminal": process.status in _TERMINAL,
            "unread_message_count": int(activity_row["unread_message_count"]),
            "interrupt_count": int(activity_row["interrupt_count"]),
            "messages": to_jsonable(activity_row["messages"] if include_messages else []),
            "llm_call_count": llm_call_count,
            "token_total": token_total,
            "resource_remaining": to_jsonable(
                resource_remaining
                if resource_remaining is not None
                else self.runtime.resources.remaining_budget(pid)
            ),
            "rating": to_jsonable(
                self.runtime.ratings.get(pid) if rating is _SUMMARY_UNSET else rating
            ),
        }

    def _process_summaries(
        self,
        *,
        limit: int,
        include_messages: bool,
        truncated: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        selected_truncated = truncated if truncated is not None else {}
        processes = _take_source_window(
            self.runtime.process.list(limit=limit + 1, active_first=True),
            limit=limit,
            path="processes",
            truncated=selected_truncated,
        )
        pids = [process.pid for process in processes]
        activity = self.runtime.store.get_process_activity_summaries(
            pids,
            recent_message_limit=(
                self.runtime.config.gui.snapshot_process_message_limit if include_messages else 0
            ),
            recent_llm_call_limit=self.runtime.config.gui.snapshot_process_llm_call_limit,
        )
        remaining = self.runtime.resources.remaining_budgets(pids)
        ratings = self.runtime.ratings.get_many(pids)
        return [
            self._process_summary(
                process.pid,
                include_messages=include_messages,
                process=process,
                activity=activity,
                resource_remaining=remaining[process.pid],
                rating=ratings.get(process.pid),
            )
            for process in processes
        ]

    def _process_summary_with_scheduler(
        self,
        pid: str,
        process: Any,
    ) -> dict[str, Any]:
        return {
            "process": self._process_summary(pid, include_messages=True, process=process),
            "scheduler": self.scheduler.status(),
        }

    def _bounded_snapshot(
        self,
        snapshot: dict[str, Any],
        *,
        source_truncated: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return _bounded_gui_payload(
            snapshot,
            string_limit=self.runtime.config.gui.snapshot_string_max_chars,
            collection_limit=self.runtime.config.gui.snapshot_collection_max_items,
            pre_truncated=source_truncated,
        )

    def _tool_summaries(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for tool in self.runtime.tools.list(limit=limit):
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

    def _register_user_llm_profiles(
        self,
        profiles: dict[str, Any],
    ) -> dict[str, Any]:
        for profile_id, profile in profiles.items():
            self.runtime.llms.register_profile(profile_id, profile)
        return dict(profiles)

    def _llm_profile_summaries(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValidationError("LLM profile list limit must be a positive integer")
        default_profile_id = self.runtime.config.llm.default_profile_id
        summaries: list[dict[str, Any]] = []
        for profile_id, profile in sorted(self.runtime.config.llm.profiles.items()):
            summaries.append(
                summarize_llm_profile(
                    profile_id,
                    profile,
                    source="config",
                    editable=False,
                    default_profile_id=default_profile_id,
                )
            )
            if limit is not None and len(summaries) >= limit:
                return summaries
        for profile_id, profile in sorted(self._user_llm_profile_cache.items()):
            summaries.append(
                summarize_llm_profile(
                    profile_id,
                    profile,
                    source="user",
                    editable=True,
                    default_profile_id=default_profile_id,
                )
            )
            if limit is not None and len(summaries) >= limit:
                break
        return summaries

    def require_llm_profile_id(self, value: Any) -> str | None:
        if value is None:
            return None
        selected = str(value).strip()
        if not selected:
            return None
        try:
            return self.runtime.llms.require_profile_id(selected)
        except ValidationError as exc:
            raise GuiServerError(HTTPStatus.BAD_REQUEST, str(exc)) from exc

    def save_user_llm_profile(
        self,
        profile_id: str,
        payload: dict[str, Any],
        *,
        preserve_omitted_fields: bool = False,
    ) -> dict[str, Any]:
        selected_id = normalize_user_llm_profile_id(profile_id)
        if selected_id in self.runtime.config.llm.profiles:
            raise GuiServerError(HTTPStatus.CONFLICT, f"config LLM profile is read-only: {selected_id}")
        selected_payload = dict(payload)
        if preserve_omitted_fields:
            existing = self._user_llm_profile_cache.get(selected_id)
            if existing is None:
                raise NotFound(f"user LLM profile not found: {selected_id}")
            selected_payload = {
                **serialize_user_llm_profile(existing),
                **selected_payload,
            }
        profile = self.user_llm_profiles.upsert(selected_id, selected_payload)
        self._user_llm_profile_cache[selected_id] = profile
        self.runtime.llms.register_profile(selected_id, profile)
        return summarize_llm_profile(
            selected_id,
            profile,
            source="user",
            editable=True,
            default_profile_id=self.runtime.config.llm.default_profile_id,
        )

    def delete_user_llm_profile(self, profile_id: str) -> dict[str, Any]:
        selected_id = normalize_user_llm_profile_id(profile_id)
        if selected_id in self.runtime.config.llm.profiles:
            raise GuiServerError(HTTPStatus.CONFLICT, f"config LLM profile is read-only: {selected_id}")
        in_use = [process.pid for process in self.runtime.process.list() if process.llm_profile_id == selected_id]
        if in_use:
            raise GuiServerError(
                HTTPStatus.CONFLICT,
                f"LLM profile is in use by existing processes: {selected_id}",
                details={"profile_id": selected_id, "pids": in_use},
            )
        self.user_llm_profiles.delete(selected_id)
        self._user_llm_profile_cache.pop(selected_id, None)
        try:
            self.runtime.llms.unregister_profile(selected_id)
        except ValidationError:
            pass
        return {"ok": True, "profile_id": selected_id}


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

    def do_PUT(self) -> None:  # noqa: N802
        self._handle("PUT")

    def do_DELETE(self) -> None:  # noqa: N802
        self._handle("DELETE")

    def _handle(self, method: str) -> None:
        parsed = urlparse(self.path)
        self._body_cached = False
        self._cached_json_body: dict[str, Any] = {}
        try:
            self._require_auth()
            if method == "GET" and parsed.path == "/api/events/stream":
                self._handle_sse(parsed)
                return
            if method in {"POST", "PUT", "DELETE"}:
                self._cached_json_body = self._read_body(optional=True)
                self._body_cached = True
            if method == "GET" and parsed.path == "/api/health":
                # Health remains non-blocking on ``runtime_lock``, but it still
                # reads the Runtime store and therefore must be drained before
                # an owned Runtime is closed.
                with self.server.service.runtime_user(serialize=False):
                    result = self._dispatch(method, parsed.path, parse_qs(parsed.query))
                should_shutdown = False
            elif _is_fast_gui_request(method, parsed.path):
                result = self._dispatch(method, parsed.path, parse_qs(parsed.query))
                should_shutdown = method == "POST" and parsed.path == "/api/shutdown"
            elif _is_object_task_wait_request(method, parsed.path):
                # Object-task wait deliberately does not serialize all GUI
                # operations, but it is still an in-flight Runtime user that
                # service shutdown must drain.
                with self.server.service.runtime_user(serialize=False):
                    result = self._dispatch(method, parsed.path, parse_qs(parsed.query))
                    should_shutdown = False
            else:
                with self.server.service.runtime_user():
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
            public_error = self.server.service.record_internal_error(
                exc,
                action="gui.request_internal_error",
                target="gui.request",
            )
            self._write_json(
                {"ok": False, "error": public_error},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _dispatch(self, method: str, path: str, query: dict[str, list[str]]) -> Any:
        service = self.server.service
        parts = [unquote(part) for part in path.strip("/").split("/") if part]
        if parts[:1] != ["api"]:
            raise GuiServerError(HTTPStatus.NOT_FOUND, "unknown endpoint")
        route = parts[1:]
        self._validate_actor_contract(method, route)
        if method == "GET" and route == ["health"]:
            return service.health()
        if method == "POST" and route == ["shutdown"]:
            if not service.shutdown():
                raise GuiServerError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "GUI runtime teardown is incomplete; retry shutdown",
                    details={"retryable": True, "status": "shutdown_incomplete"},
                )
            return {"ok": True, "status": "stopped"}
        if method == "GET" and route == ["snapshot"]:
            return service.snapshot()
        if method == "GET" and route == ["processes"]:
            limit = _bounded_query_limit(
                query,
                "limit",
                default=service.runtime.config.gui.snapshot_collection_max_items,
                maximum=service.runtime.config.gui.snapshot_collection_max_items,
            )
            return service._process_summaries(limit=limit, include_messages=True)
        if method == "GET" and route == ["operations"]:
            pid = _query_str(query, "pid")
            if not pid:
                raise GuiServerError(HTTPStatus.BAD_REQUEST, "operations requires pid")
            return service.runtime.explain.list_operations(
                pid,
                limit=_query_int(query, "limit"),
                cursor=_query_str(query, "cursor"),
            )
        if method == "GET" and route == ["operations", "resolve"]:
            kind = _query_str(query, "kind")
            evidence_id = _query_str(query, "id")
            if not kind or not evidence_id:
                raise GuiServerError(HTTPStatus.BAD_REQUEST, "operation resolve requires kind and id")
            result = service.runtime.explain.resolve(
                kind,
                evidence_id,
                evidence_limit=_query_int(query, "evidence_limit"),
                cursor=_query_str(query, "cursor"),
            )
            if result.get("ambiguous"):
                raise GuiServerError(
                    HTTPStatus.CONFLICT,
                    "operation evidence resolves to multiple causal roots",
                    details={"candidates": result.get("candidates", [])},
                )
            return result
        if method == "GET" and len(route) == 2 and route[0] == "operations":
            return service.runtime.explain.explain_operation(
                route[1],
                evidence_limit=_query_int(query, "evidence_limit"),
                cursor=_query_str(query, "cursor"),
            )
        if method == "GET" and route == ["tools"]:
            limit = _bounded_query_limit(
                query,
                "limit",
                default=service.runtime.config.gui.snapshot_collection_max_items,
                maximum=service.runtime.config.gui.snapshot_collection_max_items,
            )
            return service._tool_summaries(limit=limit)
        if method == "POST" and route == ["processes"]:
            body = self._read_body()
            max_quanta = _positive_int_or_none(body.get("max_quanta"), "max_quanta")
            llm_profile_id = service.require_llm_profile_id(body.get("llm_profile"))
            authority_manifest = body.get("authority_manifest")
            if authority_manifest is not None and not isinstance(authority_manifest, dict):
                raise GuiServerError(HTTPStatus.BAD_REQUEST, "authority_manifest must be a JSON object")
            pid = service.runtime.process.spawn(
                image=str(body["image"]) if body.get("image") is not None else None,
                goal=body.get("goal", ""),
                working_directory=body.get("working_directory"),
                llm_profile_id=llm_profile_id,
                authority_manifest=authority_manifest,
            )
            service.publish_runtime_changes("process.spawn")
            if _json_bool(body, "auto_run", True):
                service.scheduler.maybe_start(
                    max_quanta=max_quanta,
                    reason=f"spawn:{pid}",
                )
            return {"pid": pid, "process": service._process_summary(pid, include_messages=True), "scheduler": service.scheduler.status()}
        if len(route) >= 1 and route[0] == "workflows":
            return self._dispatch_workflows(method, route[1:])
        if len(route) >= 1 and route[0] == "object-tasks":
            return self._dispatch_object_tasks(method, route[1:], query)
        if route[:2] == ["scheduler", "auto"] and method == "POST":
            body = self._read_body()
            return service.scheduler.set_auto_run(_json_bool(body, "enabled", True))
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
        if len(route) >= 1 and route[0] == "llm-profiles":
            return self._dispatch_llm_profiles(method, route[1:])
        if len(route) >= 1 and route[0] == "jsonrpc":
            return self._dispatch_jsonrpc(method, route[1:], query)
        if len(route) >= 1 and route[0] == "mcp":
            return self._dispatch_mcp(method, route[1:], query)
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
            authority_manifest = body.get("authority_manifest")
            if authority_manifest is not None and not isinstance(authority_manifest, dict):
                raise GuiServerError(HTTPStatus.BAD_REQUEST, "authority_manifest must be a JSON object")
            if self._workflow_requires_confirmation(service, tool, body):
                self._require_confirmed(
                    "workflow.run",
                    body,
                    {
                        "tool": tool,
                        "image": body.get("image"),
                        "working_directory": body.get("working_directory"),
                    },
                )
            result = service.runtime.run_workflow(
                tool,
                raw_args,
                image=str(body["image"]) if body.get("image") is not None else None,
                goal=body.get("goal"),
                working_directory=str(body["working_directory"]) if body.get("working_directory") is not None else None,
                authority_manifest=authority_manifest,
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
                grant_result_to_notify=_json_bool(body, "grant_result_to_notify", False),
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
                enabled=_json_bool(body, "enabled", True),
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
            return service.human_request_views(pid=pid)
        if method == "GET" and route == ["llm-calls"]:
            limit = _bounded_query_limit(
                query,
                "limit",
                default=service.runtime.config.gui.snapshot_process_llm_call_limit,
                maximum=service.runtime.config.gui.snapshot_process_llm_call_limit,
            )
            return to_jsonable(service.runtime.store.list_llm_calls(pid=pid, limit=limit))
        if method == "GET" and route == ["rating"]:
            return to_jsonable(service.runtime.ratings.get(pid))
        if method == "POST" and route == ["rating"]:
            body = self._read_body()
            rating = service.runtime.ratings.upsert(
                pid,
                score=body.get("score"),
                comment=body.get("comment", ""),
            )
            service.publish_runtime_changes("rating.upsert")
            return to_jsonable(rating)
        if method == "GET" and route == ["audit"]:
            limit = _bounded_query_limit(
                query,
                "limit",
                default=service.runtime.config.gui.snapshot_audit_limit,
                maximum=service.runtime.config.gui.snapshot_audit_limit,
            )
            return to_jsonable(
                service.runtime.audit.trace(
                    limit=limit,
                    actor=pid,
                    target=f"process:{pid}",
                    match_any=True,
                    before_record_id=_query_str(query, "before"),
                )
            )
        if method == "GET" and route == ["events"]:
            limit = _bounded_query_limit(
                query,
                "limit",
                default=service.runtime.config.gui.snapshot_event_limit,
                maximum=service.runtime.config.gui.snapshot_event_limit,
            )
            return to_jsonable(
                service.runtime.events.list(
                    target=pid,
                    limit=limit,
                    before_event_id=_query_str(query, "before"),
                )
            )
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
            body = self._read_body(optional=True)
            service.runtime.process.resume(pid)
            service.publish_runtime_changes("process.resume")
            if _json_bool(body, "auto_run", False):
                service.scheduler.maybe_start(reason=f"resume:{pid}")
            return service._process_summary(pid, include_messages=True)
        if method == "POST" and route == ["signal"]:
            body = self._read_body()
            try:
                signal = ProcessSignal(str(body.get("signal") or ProcessSignal.INTERRUPT.value))
            except ValueError as exc:
                raise GuiServerError(HTTPStatus.BAD_REQUEST, f"unknown process signal: {body.get('signal')}") from exc
            if signal in {ProcessSignal.CANCEL, ProcessSignal.TERMINATE}:
                self._require_confirmed("process.signal", body, {"pid": pid, "signal": signal.value})
            service.runtime.process.signal(pid, signal, payload=body.get("payload"))
            service.publish_runtime_changes("process.signal")
            return service._process_summary(pid, include_messages=True)
        if method == "POST" and route in (["message"], ["interrupt"]):
            body = self._read_body()
            max_quanta = _positive_int_or_none(body.get("max_quanta"), "max_quanta")
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
            if _json_bool(body, "auto_run", True):
                service.scheduler.maybe_start(
                    max_quanta=max_quanta,
                    reason=f"message:{pid}",
                )
            return {"message": to_jsonable(message), "process": service._process_summary(pid, include_messages=True), "scheduler": service.scheduler.status()}
        if method == "POST" and route == ["cd"]:
            body = self._read_body()
            process = service.runtime.set_process_working_directory(pid, _required_body_string(body, "path"))
            service.publish_runtime_changes("process.cd")
            return service._process_summary(pid, include_messages=True, process=process)
        if method == "POST" and route == ["exec"]:
            body = self._read_body()
            max_quanta = _positive_int_or_none(body.get("max_quanta"), "max_quanta")
            llm_profile_id = service.require_llm_profile_id(body.get("llm_profile"))
            self._require_confirmed(
                "process.exec",
                body,
                {"pid": pid, "image": body.get("image"), "goal": body.get("goal"), "llm_profile": llm_profile_id},
            )
            process = service.runtime.exec_process(
                pid,
                _required_body_string(body, "image"),
                args=body.get("args") if isinstance(body.get("args"), dict) else {},
                goal=body.get("goal"),
                preserve_memory=_json_bool(body, "preserve_memory", True),
                preserve_capabilities=_json_bool(body, "preserve_capabilities", False),
                llm_profile_id=llm_profile_id,
            )
            service.publish_runtime_changes("process.exec")
            if _json_bool(body, "auto_run", True):
                service.scheduler.maybe_start(
                    max_quanta=max_quanta,
                    reason=f"exec:{pid}",
                )
            return service._process_summary_with_scheduler(pid, process)
        if method == "POST" and route == ["exit"]:
            body = self._read_body()
            self._require_confirmed("process.exit", body, {"pid": pid, "failed": body.get("failed", False)})
            service.runtime.process.exit(pid, failed=_json_bool(body, "failed", False), message=body.get("message"))
            service.publish_runtime_changes("process.exit")
            return service._process_summary(pid, include_messages=True)
        raise GuiServerError(HTTPStatus.NOT_FOUND, "unknown process endpoint")

    def _dispatch_human(self, method: str, route: list[str]) -> Any:
        service = self.server.service
        if method == "GET" and not route:
            return service.human_request_views()
        if method == "POST" and len(route) == 2 and route[1] == "respond":
            body = self._read_body()
            max_quanta = _positive_int_or_none(body.get("max_quanta"), "max_quanta")
            current = service.runtime.human.get(route[0])
            if current.status.value != "pending":
                raise GuiServerError(
                    HTTPStatus.CONFLICT,
                    f"human request is not pending: {route[0]} status={current.status.value}",
                )
            raw_decision = body.get("decision")
            if raw_decision is not None and not isinstance(raw_decision, dict):
                raise GuiServerError(HTTPStatus.BAD_REQUEST, "decision must be a JSON object")
            decision = dict(raw_decision or {})
            if "answer" in body:
                if not isinstance(body["answer"], str):
                    raise GuiServerError(HTTPStatus.BAD_REQUEST, "answer must be a string")
                decision = {**decision, "answer": body["answer"]}
            approved = _json_bool(body, "approved", False)
            request_type = current.payload.get("type")
            if request_type == "permission_request":
                policy = decision.get("policy")
                if not isinstance(policy, str) or policy not in {
                    CapabilityManager.ALWAYS_ALLOW,
                    CapabilityManager.ALWAYS_DENY,
                    CapabilityManager.ASK_EACH_TIME,
                }:
                    raise GuiServerError(
                        HTTPStatus.BAD_REQUEST,
                        "permission response decision.policy must be always_allow, always_deny, or ask_each_time",
                    )
            if request_type == "question" and approved and not isinstance(decision.get("answer"), str):
                raise GuiServerError(HTTPStatus.BAD_REQUEST, "approved question response requires a string answer")
            if approved:
                request = service.runtime.human.approve_for_presentation(
                    route[0],
                    presentation="gui",
                    decision={"approved": True, "source": "gui", **decision},
                )
            else:
                request = service.runtime.human.reject_for_presentation(
                    route[0],
                    presentation="gui",
                    decision={"approved": False, "source": "gui", **decision},
                )
            service.publish_runtime_changes("human.respond")
            if _json_bool(body, "auto_run", True):
                service.scheduler.maybe_start(max_quanta=max_quanta, reason=f"human:{route[0]}")
            return {
                "request": service.human_request_view(request),
                "scheduler": service.scheduler.status(),
            }
        raise GuiServerError(HTTPStatus.NOT_FOUND, "unknown human endpoint")

    def _dispatch_checkpoints(self, method: str, route: list[str], query: dict[str, list[str]]) -> Any:
        service = self.server.service
        if method == "GET" and not route:
            return service.runtime.checkpoint.list(pid=_query_str(query, "pid"), actor=None, require_capability=False)
        if method == "POST" and route == ["create"]:
            body = self._read_body()
            checkpoint_id = service.runtime.checkpoint.create(
                _required_body_string(body, "pid"),
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
            actor_value = body.get("actor")
            replace = _json_bool(body, "replace", False)
            if actor_value is not None:
                result = service.runtime.skills.register_skill_from_workspace_path(
                    str(actor_value),
                    _required_body_string(body, "path"),
                    replace=replace,
                    require_capability=True,
                )
            else:
                raise GuiServerError(
                    HTTPStatus.BAD_REQUEST,
                    "GUI skill path registration requires an actor and workspace filesystem authority",
                )
            service.publish_runtime_changes("skill.register")
            return result
        if method == "POST" and len(route) == 2 and route[1] in {"activate", "unload"}:
            body = self._read_body()
            expected_package_sha256 = (
                _required_package_sha256(body, "expected_package_sha256")
                if route[1] == "activate"
                else None
            )
            self._require_confirmed(
                f"skill.{route[1]}",
                body,
                {
                    "pid": body.get("pid"),
                    "skill_id": route[0],
                    **(
                        {"expected_package_sha256": expected_package_sha256}
                        if expected_package_sha256 is not None
                        else {}
                    ),
                    "admin_mode": body.get("actor") is None,
                },
            )
            require_capability = body.get("actor") is not None
            actor = str(body.get("actor") or "gui")
            if route[1] == "activate":
                result = service.runtime.skills.activate_skill(
                    _required_body_string(body, "pid"),
                    route[0],
                    actor=actor,
                    require_capability=require_capability,
                    expected_package_sha256=expected_package_sha256,
                )
            else:
                result = service.runtime.skills.unload_skill(
                    _required_body_string(body, "pid"),
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
            return self._list_capabilities(query)
        if method == "GET" and len(route) == 1:
            return service.runtime.capability.inspect(route[0])
        if method == "POST" and route == ["grant"]:
            body = self._read_body()
            self._require_confirmed("capability.grant", body, {"subject": body.get("subject"), "resource": body.get("resource"), "rights": body.get("rights")})
            rights = _gui_capability_rights(body.get("rights"))
            actor = body.get("actor")
            if actor is None:
                cap = service.runtime.capability.grant(
                    _required_body_string(body, "subject"),
                    _required_body_string(body, "resource"),
                    rights,
                    issued_by="gui",
                )
            else:
                cap = service.runtime.capability.issue(
                    actor=str(actor),
                    subject=_required_body_string(body, "subject"),
                    spec=CapabilitySpec(resource=_required_body_string(body, "resource"), rights=set(rights)),
                    require_authority=True,
                )
            service.publish_runtime_changes("capability.grant")
            return to_jsonable(cap)
        if method == "POST" and route == ["delegate"]:
            body = self._read_body()
            self._require_confirmed("capability.delegate", body, {"parent": body.get("parent"), "child": body.get("child"), "resource": body.get("resource"), "rights": body.get("rights")})
            actor = body.get("actor")
            parent = _required_body_string(body, "parent")
            if actor is not None and parent != str(actor):
                raise CapabilityDenied("GUI actor-mode delegation may only delegate from the actor process")
            cap = service.runtime.capability.delegate(
                parent,
                _required_body_string(body, "child"),
                {"resource": _required_body_string(body, "resource"), "rights": _gui_capability_rights(body.get("rights"))},
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
            return service.runtime.capability.explain_decision(
                _required_body_string(body, "subject"),
                _required_body_string(body, "resource"),
                _required_body_string(body, "right"),
            )
        raise GuiServerError(HTTPStatus.NOT_FOUND, "unknown capability endpoint")

    def _list_capabilities(self, query: dict[str, list[str]]) -> Any:
        runtime = self.server.service.runtime
        subject = _query_str(query, "subject")
        configured_limit = runtime.config.capability.list_limit
        limit = _bounded_query_limit(
            query,
            "limit",
            default=configured_limit,
            maximum=configured_limit,
        )
        if _query_str(query, "mode") == "page":
            page = runtime.capability.presentation_page(
                subject=subject,
                include_inactive=True,
                limit=limit,
                after_cap_id=_query_str(query, "after"),
                max_bytes=runtime.config.gui.sse_payload_max_bytes,
            )
            return to_jsonable(
                {
                    "items": page.capabilities,
                    "next_after": page.next_cursor,
                    "has_more": page.has_more,
                }
            )
        capabilities = (
            runtime.capability.list_subject(
                subject,
                include_inactive=True,
                limit=limit,
            )
            if subject
            else runtime.store.list_capabilities(limit=limit)
        )
        return to_jsonable(capabilities)

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
                replace=_json_bool(body, "replace", False),
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
                "required_modules_count": len(result.image.required_modules),
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
                checkpoint_id=_required_body_string(body, "checkpoint_id"),
                image_id=_required_body_string(body, "image_id"),
                name=_required_body_string(body, "name"),
                version=str(body.get("version") or "v0"),
                replace=_json_bool(body, "replace", False),
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
                "required_modules_count": len(result.image.required_modules),
            }
        raise GuiServerError(HTTPStatus.NOT_FOUND, "unknown image endpoint")

    def _dispatch_llm_profiles(self, method: str, route: list[str]) -> Any:
        service = self.server.service
        if method == "GET" and not route:
            return service._llm_profile_summaries()
        if method == "POST" and not route:
            body = self._read_body()
            profile_id = str(body.get("profile_id") or "").strip()
            if not profile_id:
                raise GuiServerError(HTTPStatus.BAD_REQUEST, "profile_id is required")
            summary = service.save_user_llm_profile(profile_id, body)
            service.publish_runtime_changes("llm_profile.upsert")
            return summary
        if len(route) == 1 and method == "PUT":
            body = self._read_body()
            summary = service.save_user_llm_profile(
                route[0],
                body,
                preserve_omitted_fields=True,
            )
            service.publish_runtime_changes("llm_profile.upsert")
            return summary
        if len(route) == 1 and method == "DELETE":
            result = service.delete_user_llm_profile(route[0])
            service.publish_runtime_changes("llm_profile.delete")
            return result
        raise GuiServerError(HTTPStatus.NOT_FOUND, "unknown LLM profile endpoint")

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
                replace=_json_bool(body, "replace", False),
                require_capability=body.get("actor") is not None,
                source=body.get("source"),
            )
            service.publish_runtime_changes("jsonrpc.register")
            return result
        if method == "POST" and len(route) == 2 and route[1] == "call":
            body = self._read_body()
            self._require_confirmed("jsonrpc.call", body, {"pid": body.get("pid"), "endpoint_id": route[0], "method_id": body.get("method_id")})
            result = service.runtime.jsonrpc.call(
                _required_body_string(body, "pid"),
                route[0],
                _required_body_string(body, "method_id"),
                params=body.get("params"),
            )
            service.publish_runtime_changes("jsonrpc.call")
            return to_jsonable(result)
        raise GuiServerError(HTTPStatus.NOT_FOUND, "unknown JSON-RPC endpoint")

    def _dispatch_mcp(self, method: str, route: list[str], query: dict[str, list[str]]) -> Any:
        service = self.server.service
        if method == "GET" and not route:
            return service.runtime.mcp.list_servers(text=_query_str(query, "text"), require_capability=False)
        if method == "GET" and len(route) == 1:
            return service.runtime.mcp.inspect_server(route[0], require_capability=False)
        if method == "GET" and len(route) == 2 and route[1] == "tools":
            refresh_value = (_query_str(query, "refresh") or "").lower()
            return service.runtime.mcp.list_tools(
                route[0],
                actor="gui",
                require_capability=False,
                refresh=refresh_value in {"1", "true", "yes", "on"},
            )
        if method == "POST" and route == ["register"]:
            body = self._read_body()
            self._require_confirmed("mcp.register", body, {"source": body.get("source")})
            if "path" in body:
                raise GuiServerError(
                    HTTPStatus.BAD_REQUEST,
                    "GUI MCP registration accepts manifest_text, not host file paths",
                )
            text = body.get("manifest_text")
            if not isinstance(text, str) or not text.strip():
                raise GuiServerError(HTTPStatus.BAD_REQUEST, "MCP registration requires non-empty manifest_text")
            result = service.runtime.mcp.register_server_from_yaml_text(
                text,
                actor=str(body.get("actor") or "gui"),
                replace=_json_bool(body, "replace", False),
                require_capability=body.get("actor") is not None,
                source=body.get("source"),
            )
            service.publish_runtime_changes("mcp.register")
            return result
        if method == "POST" and len(route) == 2 and route[1] == "call":
            body = self._read_body()
            self._require_confirmed(
                "mcp.call",
                body,
                {"pid": body.get("pid"), "server_id": route[0], "tool_id": body.get("tool_id")},
            )
            result = service.runtime.mcp.call_tool(
                _required_body_string(body, "pid"),
                route[0],
                _required_body_string(body, "tool_id"),
                arguments=body["arguments"] if "arguments" in body and body["arguments"] is not None else {},
            )
            service.publish_runtime_changes("mcp.call")
            return to_jsonable(result)
        raise GuiServerError(HTTPStatus.NOT_FOUND, "unknown MCP endpoint")

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
            while not self.server.service.closed:
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
        event_name, payload_data = _sse_payload_data(
            event.event,
            event.data,
            max_bytes=self.server.service.runtime.config.gui.sse_payload_max_bytes,
            string_limit=self.server.service.runtime.config.gui.snapshot_string_max_chars,
            collection_limit=self.server.service.runtime.config.gui.snapshot_collection_max_items,
        )
        payload = json.dumps(payload_data, ensure_ascii=False, default=str)
        self.wfile.write(f"id: {event.seq}\nevent: {event_name}\ndata: {payload}\n\n".encode("utf-8"))
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
            value = bounded_json_loads(self.rfile.read(length))
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise GuiServerError(HTTPStatus.BAD_REQUEST, f"invalid JSON body: {exc}") from exc
        if not isinstance(value, dict):
            raise GuiServerError(HTTPStatus.BAD_REQUEST, "request body must be a JSON object")
        _validate_json_bool_fields(value)
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

    def _validate_actor_contract(
        self,
        method: str,
        route: list[str],
    ) -> None:
        if method not in {"POST", "PUT", "DELETE"}:
            return
        body = self._read_body(optional=True)
        if "actor" not in body:
            return
        if not _gui_route_accepts_actor(method, route):
            target = "/api/" + "/".join(route)
            raise GuiServerError(
                HTTPStatus.BAD_REQUEST,
                f"{method} {target} does not accept actor; this endpoint uses Host/admin or its explicit pid field",
            )
        _required_body_string(body, "actor")

    def _workflow_requires_confirmation(self, service: GuiRuntimeService, tool: str, body: dict[str, Any]) -> bool:
        if body.get("image") is not None or body.get("working_directory") is not None:
            return True
        try:
            handle = service.runtime.tools.resolve(tool)
        except NotFound:
            return True
        return service.runtime.tools.has_side_effects(handle)

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
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        if self.close_connection:
            self.send_header("Connection", "close")


def create_gui_http_server(
    *,
    db: str | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
    token: str | None = None,
    auto_run: bool = True,
    max_quanta: int | None | object = _CONFIG_DEFAULT,
    runtime: Runtime | None = None,
    config: AgentLibOSConfig | None = None,
    llm_profiles_file: str | Path | None = None,
) -> GuiHTTPServer:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("GUI server is local-only and must bind 127.0.0.1")
    service = GuiRuntimeService(
        db=db,
        runtime=runtime,
        config=config,
        token=token,
        auto_run=auto_run,
        max_quanta=max_quanta,
        llm_profiles_file=llm_profiles_file,
    )
    try:
        return GuiHTTPServer(("127.0.0.1", int(port)), service)
    except BaseException as bind_error:
        try:
            _shutdown_gui_service_before_exit(service)
        except BaseException as shutdown_error:
            bind_error.add_note(
                "GUI service teardown also failed after server bind failure: "
                f"{type(shutdown_error).__name__}: {shutdown_error}"
            )
        raise


def serve(
    *,
    db: str | None = None,
    port: int,
    token: str | None,
    auto_run: bool,
    max_quanta: int | None | object,
    config: AgentLibOSConfig | None = None,
    llm_profiles_file: str | Path | None = None,
    ready: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    server = create_gui_http_server(
        db=db,
        port=port,
        token=token,
        auto_run=auto_run,
        max_quanta=max_quanta,
        config=config,
        llm_profiles_file=llm_profiles_file,
    )
    try:
        host, selected_port = server.server_address
        payload = {"url": f"http://{host}:{selected_port}", "token": server.service.token, "db": server.service.db}
        if ready is not None:
            ready(payload)
        else:
            print(json.dumps(payload, ensure_ascii=True), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            # Ctrl-C is the normal foreground-server shutdown path. The
            # finally block still performs the bounded Runtime teardown.
            pass
    finally:
        try:
            _shutdown_gui_service_before_exit(server.service)
        finally:
            server.server_close()


def _shutdown_gui_service_before_exit(service: GuiRuntimeService, *, attempts: int = 2) -> None:
    """Finish owned Runtime teardown or make process exit fail visibly."""

    selected_attempts = max(1, int(attempts))
    failures: list[str] = []
    last_error: Exception | None = None
    for _attempt in range(selected_attempts):
        try:
            if service.shutdown():
                return
            failures.append("shutdown returned false")
        except Exception as exc:
            last_error = exc
            failures.append(f"{type(exc).__name__}: {exc}")
    error = RuntimeError(
        f"GUI runtime teardown remained incomplete after {selected_attempts} attempts: "
        + "; ".join(failures)
    )
    if last_error is not None:
        raise error from last_error
    raise error


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agent-libos-gui-server")
    parser.add_argument("--config", help="YAML config overlay. Defaults to the project-root config.yaml when present.")
    parser.add_argument("--db")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--token")
    parser.add_argument(
        "--llm-profiles-file",
        default=None,
        help=f"User-level GUI LLM profile JSON file. Defaults to {default_user_llm_profiles_path()}.",
    )
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
    selected_config = _load_runtime_config(args.config, parser)
    serve(
        db=args.db,
        port=args.port,
        token=args.token,
        auto_run=not args.no_auto_run,
        max_quanta=args.max_quanta,
        config=selected_config,
        llm_profiles_file=args.llm_profiles_file,
    )


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            return int(stripped, 10)
        except ValueError as exc:
            raise GuiServerError(HTTPStatus.BAD_REQUEST, "integer value expected") from exc
    if isinstance(value, bool):
        raise GuiServerError(HTTPStatus.BAD_REQUEST, "integer value expected")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        raise GuiServerError(HTTPStatus.BAD_REQUEST, "integer value expected")
    raise GuiServerError(HTTPStatus.BAD_REQUEST, "integer value expected")


def _positive_int_or_none(value: Any, name: str) -> int | None:
    try:
        parsed = _int_or_none(value)
    except GuiServerError as exc:
        raise GuiServerError(HTTPStatus.BAD_REQUEST, f"{name} must be a positive integer or omitted") from exc
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


def _bounded_query_limit(
    query: dict[str, list[str]],
    key: str,
    *,
    default: int,
    maximum: int,
) -> int:
    selected = _query_int(query, key)
    if selected is None:
        return default
    if selected <= 0:
        raise GuiServerError(HTTPStatus.BAD_REQUEST, f"{key} must be a positive integer")
    if selected > maximum:
        raise GuiServerError(HTTPStatus.BAD_REQUEST, f"{key} must be at most {maximum}")
    return selected


def _query_str(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values else None


def _is_object_task_wait_request(method: str, path: str) -> bool:
    if method != "POST":
        return False
    parts = [unquote(part) for part in path.strip("/").split("/") if part]
    return len(parts) == 4 and parts[:2] == ["api", "object-tasks"] and parts[3] == "wait"


def _is_fast_gui_request(method: str, path: str) -> bool:
    parts = [unquote(part) for part in path.strip("/").split("/") if part]
    if parts == ["api", "health"] and method == "GET":
        return True
    if parts == ["api", "shutdown"] and method == "POST":
        return True
    if parts == ["api", "scheduler", "pause"] and method == "POST":
        return True
    if parts == ["api", "scheduler", "auto"] and method == "POST":
        return True
    return False


def _gui_route_accepts_actor(method: str, route: list[str]) -> bool:
    """Return whether a mutation actually applies process-scoped authority.

    ``actor`` is security-sensitive: accepting it on a Host/admin endpoint while
    silently ignoring it would let a caller mistake attribution for an
    authorization boundary. Keep this allowlist next to request dispatch rather
    than inferring support from arbitrary request fields.
    """

    if method != "POST":
        return False
    if tuple(route) in {
        ("checkpoints", "create"),
        ("skills", "register"),
        ("capabilities", "grant"),
        ("capabilities", "delegate"),
        ("images", "register"),
        ("images", "commit"),
        ("jsonrpc", "register"),
        ("mcp", "register"),
    }:
        return True
    if len(route) == 3 and route[0] == "checkpoints" and route[2] in {"restore", "fork"}:
        return True
    if len(route) == 3 and route[0] == "skills" and route[2] in {"activate", "unload"}:
        return True
    return len(route) == 3 and route[0] == "capabilities" and route[2] == "revoke"


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
    enabled = _json_bool(body, "owner_watch", False) or bool(events or body.get("watch_channel") or "watch_kind" in body)
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
    if origin == _GUI_PRODUCTION_RENDERER_ORIGIN:
        return origin
    parsed = urlparse(origin)
    if parsed.scheme in {"http", "https"} and parsed.hostname in {"127.0.0.1", "localhost"}:
        return origin
    return None
