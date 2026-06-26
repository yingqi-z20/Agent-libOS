from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from dataclasses import replace
from typing import Any, TYPE_CHECKING

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models.exceptions import (
    HumanApprovalRequired,
    ProcessMessageWaitRequired,
    ProcessWaitRequired,
    ResourceLimitExceeded,
)
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.llm.action_parser import parse_json_action
from agent_libos.llm.client import LLMClient
from agent_libos.llm.context_memory import LLMContextMemory
from agent_libos.llm.prompt import build_system_prompt, build_user_prompt
from agent_libos.llm.records import observable_llm_call_fields
from agent_libos.llm.tool_protocol import tool_call_to_action
from agent_libos.tools.observability import sanitize_for_observability
from agent_libos.models import (
    EventType,
    HumanRequestStatus,
    LLMCallRecord,
    ObjectHandle,
    ObjectRight,
    ProcessMessageKind,
    ProcessStatus,
    ResourceUsage,
    ViewMode,
)

if TYPE_CHECKING:
    from agent_libos.runtime.runtime import Runtime


class LLMProcessExecutor:
    """Runs one model-selected tool action per process quantum."""

    def __init__(self, runtime: "Runtime", client: LLMClient | None = None, config: AgentLibOSConfig | None = None):
        self.runtime = runtime
        self.config = config or DEFAULT_CONFIG
        if client is not None:
            self.runtime.llms.set_test_client(self.config.llm.default_profile_id, client)
        # Pending actions are held outside Object Memory because the process has
        # not received a tool result yet. The action is retried after the human
        # queue records a decision, without asking the model for a new action.
        self._pending_human_actions: dict[str, dict[str, Any]] = {}
        self._pending_wait_actions: dict[str, dict[str, Any]] = {}
        self._pending_message_actions: dict[str, dict[str, Any]] = {}
        self.context_memory = LLMContextMemory(runtime)
        self._load_pending_actions()

    @property
    def client(self) -> Any:
        """Compatibility view of the default LLM profile client."""
        return self.runtime.llms.default_client

    @client.setter
    def client(self, value: Any) -> None:
        self.runtime.llms.set_test_client(self.config.llm.default_profile_id, value)

    def run_once(self, pid: str) -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun_once(pid))
        raise RuntimeError("Cannot call run_once() inside a running event loop. Use await arun_once(...).")

    async def arun_once(self, pid: str) -> dict[str, Any]:
        process = self.runtime.process.get(pid)
        if process.status not in {ProcessStatus.RUNNING, ProcessStatus.RUNNABLE}:
            return {"ok": False, "skipped": True, "status": process.status.value}
        if pid in self._pending_human_actions:
            return await self._resume_pending_human_action(pid)
        if pid in self._pending_wait_actions:
            return await self._resume_pending_wait_action(pid)
        if pid in self._pending_message_actions:
            return await self._resume_pending_message_action(pid)
        image = self.runtime.images.get(process.image_id)
        if image is None:
            error = f"agent image not found for process {pid}: {process.image_id}"
            self.runtime.process.exit(pid, failed=True, message=error)
            self.runtime.audit.record(
                actor=pid,
                action="llm.image_missing",
                target=f"image:{process.image_id}",
                decision={"error": error},
            )
            return {"ok": False, "error": error}
        if process.memory_view is None:
            process.memory_view = self.runtime.memory.create_view(pid, [], mode=ViewMode.READ_ONLY)
            process.updated_at = utc_now()
            self.runtime.store.update_process(process)

        self._notify_interrupt_messages(pid)
        source_view = self.context_memory.view_without_context(pid, process.memory_view)
        source_context = self.runtime.memory.materialize_context(
            pid,
            source_view,
            policy=image.context_policy,
            budget_tokens=process.resource_budget.max_context_materialization_tokens,
            charge_resources=False,
        )
        events = [
            replace(
                event,
                source=self.runtime.tools.redact_model_context(pid, event.source),
                payload=self.runtime.tools.redact_model_context(pid, event.payload),
                correlation_id=self.runtime.tools.redact_model_context(pid, event.correlation_id),
                causality=self.runtime.tools.redact_model_context(pid, event.causality),
            )
            for event in self.runtime.events.list(target=pid)
        ]
        capabilities = self.runtime.capability.capabilities_for(pid)
        # The prompt-visible tool list must match the process tool table. The
        # broker still owns the real execute check, but showing extra tools
        # teaches the model to choose actions the process cannot call.
        tools = self.runtime.tools.model_visible_tools(pid)
        prompt_process = replace(
            process,
            tool_table=self.runtime.tools.model_tool_table(pid),
            loaded_skills=self.runtime.tools.model_loaded_skills(pid),
        )
        skills = self.runtime.skills.prompt_context(pid)
        try:
            context = self.context_memory.prepare(
                pid=pid,
                image=image,
                process=prompt_process,
                source_context=source_context,
                events=events,
                capabilities=capabilities,
                tools=tools,
            )
        except ResourceLimitExceeded as exc:
            self.runtime.resources.kill_if_exceeded(pid, reason=str(exc))
            self.runtime.audit.record(
                actor=pid,
                action="llm.resource_limit_exceeded",
                target=f"process:{pid}",
                decision={"error": str(exc)},
            )
            return {"ok": False, "resource_limit_exceeded": True, "error": str(exc)}
        messages = [
            {"role": "system", "content": build_system_prompt(image)},
            {
                "role": "user",
                "content": build_user_prompt(
                    process=prompt_process,
                    context=context,
                    events=events,
                    capabilities=capabilities,
                    tools=tools,
                    skills=skills,
                    prompt_mode=image.prompt_mode,
                ),
            },
        ]
        self.runtime.audit.record(
            actor=pid,
            action="llm.request",
            target=f"image:{image.image_id}",
            input_refs=context.object_refs,
            decision={"messages": len(messages), "policy": image.context_policy},
        )
        try:
            completion, action = await self._complete_valid_action(
                pid,
                messages,
                self.runtime.tools.openai_tool_schemas(pid),
            )
            try:
                result = await self.adispatch(pid, action)
            except HumanApprovalRequired as exc:
                return self._wait_for_human_action(
                    pid=pid,
                    action=action,
                    request_id=exc.request_id,
                    message=str(exc),
                    content_preview=completion.content[: self.config.llm.content_preview_chars],
                    tool_call_count=len(completion.tool_calls),
                )
            except ProcessWaitRequired as exc:
                return self._wait_for_child_action(
                    pid=pid,
                    action=exc.resume_action or action,
                    child_pid=exc.child_pid,
                    message=str(exc),
                    content_preview=completion.content[: self.config.llm.content_preview_chars],
                    tool_call_count=len(completion.tool_calls),
                )
            except ProcessMessageWaitRequired as exc:
                return self._wait_for_message_action(
                    pid=pid,
                    action=action,
                    filters=exc.filters,
                    message=str(exc),
                    content_preview=completion.content[: self.config.llm.content_preview_chars],
                    tool_call_count=len(completion.tool_calls),
                )
            return self._completed_action_result(
                pid=pid,
                action=action,
                result=result,
                content_preview=completion.content[: self.config.llm.content_preview_chars],
                tool_call_count=len(completion.tool_calls),
            )
        except HumanApprovalRequired as exc:
            self.runtime.audit.record(
                actor=pid,
                action="llm.action_waiting_human",
                target=f"human_request:{exc.request_id}",
                decision={"request_id": exc.request_id, "message": str(exc)},
            )
            return {"ok": False, "waiting_human": True, "request_id": exc.request_id}
        except ProcessWaitRequired as exc:
            self.runtime.audit.record(
                actor=pid,
                action="llm.action_waiting_child",
                target=f"process:{exc.child_pid}",
                decision={"child_pid": exc.child_pid, "message": str(exc)},
            )
            return {"ok": False, "waiting_event": True, "child_pid": exc.child_pid}
        except ProcessMessageWaitRequired as exc:
            self.runtime.audit.record(
                actor=pid,
                action="llm.action_waiting_message",
                target=f"process:{pid}",
                decision={"recipient_pid": exc.recipient_pid, "filters": exc.filters, "message": str(exc)},
            )
            return {"ok": False, "waiting_message": True, "filters": exc.filters}
        except ResourceLimitExceeded as exc:
            self.runtime.resources.kill_if_exceeded(pid, reason=str(exc))
            self.runtime.audit.record(
                actor=pid,
                action="llm.resource_limit_exceeded",
                target=f"process:{pid}",
                decision={"error": str(exc)},
            )
            return {"ok": False, "resource_limit_exceeded": True, "error": str(exc)}
        except Exception as exc:
            self.runtime.process.exit(pid, failed=True, message=f"LLM quantum failed: {exc}")
            self.runtime.audit.record(
                actor=pid,
                action="llm.action_failed",
                target=f"process:{pid}",
                decision={"error": str(exc)},
            )
            return {"ok": False, "error": str(exc)}

    def _completed_action_result(
        self,
        pid: str,
        action: dict[str, Any],
        result: dict[str, Any],
        content_preview: str,
        tool_call_count: int,
        resumed_after_human: bool = False,
        resumed_after_message: bool = False,
    ) -> dict[str, Any]:
        self.runtime.audit.record(
            actor=pid,
            action="llm.action",
            target=action.get("action"),
            decision={
                "action": sanitize_for_observability(action),
                "result": sanitize_for_observability(result),
                "content_preview": content_preview,
                "tool_call_count": tool_call_count,
                "resumed_after_human": resumed_after_human,
                "resumed_after_message": resumed_after_message,
            },
        )
        payload = {"ok": True, "action": action, "result": result}
        if resumed_after_human:
            payload["resumed_after_human"] = True
        if resumed_after_message:
            payload["resumed_after_message"] = True
        return payload

    def _wait_for_human_action(
        self,
        pid: str,
        action: dict[str, Any],
        request_id: str,
        message: str,
        content_preview: str,
        tool_call_count: int,
    ) -> dict[str, Any]:
        self._pending_human_actions[pid] = {
            "request_id": request_id,
            "action": dict(action),
            "content_preview": content_preview,
            "tool_call_count": tool_call_count,
        }
        self._persist_pending_action(
            pid,
            wait_type="human",
            request_id=request_id,
            action=action,
            content_preview=content_preview,
            tool_call_count=tool_call_count,
        )
        self.runtime.audit.record(
            actor=pid,
            action="llm.action_waiting_human",
            target=f"human_request:{request_id}",
            decision={
                "request_id": request_id,
                "action": sanitize_for_observability(action),
                "message": message,
                "tool_call_count": tool_call_count,
            },
        )
        return {"ok": False, "waiting_human": True, "request_id": request_id}

    def _wait_for_child_action(
        self,
        pid: str,
        action: dict[str, Any],
        child_pid: str,
        message: str,
        content_preview: str,
        tool_call_count: int,
    ) -> dict[str, Any]:
        self._pending_wait_actions[pid] = {
            "child_pid": child_pid,
            "action": dict(action),
            "content_preview": content_preview,
            "tool_call_count": tool_call_count,
        }
        self._persist_pending_action(
            pid,
            wait_type="child",
            child_pid=child_pid,
            action=action,
            content_preview=content_preview,
            tool_call_count=tool_call_count,
        )
        self.runtime.audit.record(
            actor=pid,
            action="llm.action_waiting_child",
            target=f"process:{child_pid}",
            decision={
                "child_pid": child_pid,
                "action": sanitize_for_observability(action),
                "message": message,
                "tool_call_count": tool_call_count,
            },
        )
        return {"ok": False, "waiting_event": True, "child_pid": child_pid}

    def _wait_for_message_action(
        self,
        pid: str,
        action: dict[str, Any],
        filters: dict[str, Any],
        message: str,
        content_preview: str,
        tool_call_count: int,
    ) -> dict[str, Any]:
        self._pending_message_actions[pid] = {
            "filters": dict(filters),
            "action": dict(action),
            "content_preview": content_preview,
            "tool_call_count": tool_call_count,
        }
        self._persist_pending_action(
            pid,
            wait_type="message",
            filters=filters,
            action=action,
            content_preview=content_preview,
            tool_call_count=tool_call_count,
        )
        self.runtime.audit.record(
            actor=pid,
            action="llm.action_waiting_message",
            target=f"process:{pid}",
            decision={
                "filters": filters,
                "action": sanitize_for_observability(action),
                "message": message,
                "tool_call_count": tool_call_count,
            },
        )
        return {"ok": False, "waiting_message": True, "filters": filters}

    async def _resume_pending_human_action(self, pid: str) -> dict[str, Any]:
        pending = self._pending_human_actions[pid]
        request_id = str(pending["request_id"])
        request = self.runtime.human.get(request_id)
        if request.status == HumanRequestStatus.PENDING:
            return {"ok": False, "waiting_human": True, "request_id": request_id}

        action = dict(pending["action"])
        self._pending_human_actions.pop(pid, None)
        if request.status == HumanRequestStatus.APPROVED or (
            self._action_name(action) == "request_permission" and request.status == HumanRequestStatus.REJECTED
        ):
            # Re-dispatch the exact same action. The resume request id is scoped
            # to this single tool call, so concurrent tool calls cannot observe
            # another process' human decision.
            try:
                result = await self.adispatch(
                    pid,
                    action,
                    context_metadata={"human_resume_request_id": request_id},
                )
            except HumanApprovalRequired as exc:
                return self._wait_for_human_action(
                    pid=pid,
                    action=action,
                    request_id=exc.request_id,
                    message=str(exc),
                    content_preview=str(pending.get("content_preview", "")),
                    tool_call_count=int(pending.get("tool_call_count", 0)),
                )
            except ProcessMessageWaitRequired as exc:
                return self._wait_for_message_action(
                    pid=pid,
                    action=action,
                    filters=exc.filters,
                    message=str(exc),
                    content_preview=str(pending.get("content_preview", "")),
                    tool_call_count=int(pending.get("tool_call_count", 0)),
                )
            except ProcessWaitRequired as exc:
                return self._wait_for_child_action(
                    pid=pid,
                    action=exc.resume_action or action,
                    child_pid=exc.child_pid,
                    message=str(exc),
                    content_preview=str(pending.get("content_preview", "")),
                    tool_call_count=int(pending.get("tool_call_count", 0)),
                )
            self._clear_pending_action(pid)
            return self._completed_action_result(
                pid=pid,
                action=action,
                result=result,
                content_preview=str(pending.get("content_preview", "")),
                tool_call_count=int(pending.get("tool_call_count", 0)),
                resumed_after_human=True,
            )

        error = f"human rejected approval request {request_id}"
        # A rejected per-use approval is surfaced as a failed action result, not
        # as a runtime crash, so the process can explain or choose another path.
        self._emit_pending_action_rejected(pid, action, request_id, error)
        result = {"ok": False, "tool_id": None, "result_oid": None, "payload": None, "error": error}
        self._clear_pending_action(pid)
        return self._completed_action_result(
            pid=pid,
            action=action,
            result=result,
            content_preview=str(pending.get("content_preview", "")),
            tool_call_count=int(pending.get("tool_call_count", 0)),
            resumed_after_human=True,
        )

    def _action_name(self, action: dict[str, Any]) -> str:
        return str(action.get("action") or action.get("tool") or action.get("name") or "")

    async def _resume_pending_wait_action(self, pid: str) -> dict[str, Any]:
        pending = self._pending_wait_actions[pid]
        child_pid = str(pending["child_pid"])
        child = self.runtime.process.get(child_pid)
        if child.status not in {ProcessStatus.EXITED, ProcessStatus.FAILED, ProcessStatus.KILLED}:
            return {"ok": False, "waiting_event": True, "child_pid": child_pid}

        action = dict(pending["action"])
        self._pending_wait_actions.pop(pid, None)
        try:
            result = await self.adispatch(
                pid,
                action,
                context_metadata={
                    "pending_child_resume": True,
                    "pending_child_pid": child_pid,
                },
            )
        except ProcessWaitRequired as exc:
            return self._wait_for_child_action(
                pid=pid,
                action=exc.resume_action or action,
                child_pid=exc.child_pid,
                message=str(exc),
                content_preview=str(pending.get("content_preview", "")),
                tool_call_count=int(pending.get("tool_call_count", 0)),
            )
        except HumanApprovalRequired as exc:
            return self._wait_for_human_action(
                pid=pid,
                action=action,
                request_id=exc.request_id,
                message=str(exc),
                content_preview=str(pending.get("content_preview", "")),
                tool_call_count=int(pending.get("tool_call_count", 0)),
            )
        except ProcessMessageWaitRequired as exc:
            return self._wait_for_message_action(
                pid=pid,
                action=action,
                filters=exc.filters,
                message=str(exc),
                content_preview=str(pending.get("content_preview", "")),
                tool_call_count=int(pending.get("tool_call_count", 0)),
            )
        self._clear_pending_action(pid)
        return self._completed_action_result(
            pid=pid,
            action=action,
            result=result,
            content_preview=str(pending.get("content_preview", "")),
            tool_call_count=int(pending.get("tool_call_count", 0)),
            resumed_after_human=False,
        )

    async def _resume_pending_message_action(self, pid: str) -> dict[str, Any]:
        pending = self._pending_message_actions[pid]
        filters = dict(pending.get("filters") or {})
        messages = self.runtime.messages.unread(
            pid,
            kind=filters.get("kind"),
            sender=filters.get("sender"),
            channel=filters.get("channel"),
            correlation_id=filters.get("correlation_id"),
            reply_to=filters.get("reply_to"),
            message_ids=filters.get("message_ids"),
        )
        if not messages:
            return {"ok": False, "waiting_message": True, "filters": filters}
        action = dict(pending["action"])
        self._pending_message_actions.pop(pid, None)
        try:
            result = await self.adispatch(pid, action)
        except ProcessMessageWaitRequired as exc:
            return self._wait_for_message_action(
                pid=pid,
                action=action,
                filters=exc.filters,
                message=str(exc),
                content_preview=str(pending.get("content_preview", "")),
                tool_call_count=int(pending.get("tool_call_count", 0)),
            )
        except ProcessWaitRequired as exc:
            return self._wait_for_child_action(
                pid=pid,
                action=exc.resume_action or action,
                child_pid=exc.child_pid,
                message=str(exc),
                content_preview=str(pending.get("content_preview", "")),
                tool_call_count=int(pending.get("tool_call_count", 0)),
            )
        except HumanApprovalRequired as exc:
            return self._wait_for_human_action(
                pid=pid,
                action=action,
                request_id=exc.request_id,
                message=str(exc),
                content_preview=str(pending.get("content_preview", "")),
                tool_call_count=int(pending.get("tool_call_count", 0)),
            )
        self._clear_pending_action(pid)
        completed = self._completed_action_result(
            pid=pid,
            action=action,
            result=result,
            content_preview=str(pending.get("content_preview", "")),
            tool_call_count=int(pending.get("tool_call_count", 0)),
            resumed_after_message=True,
        )
        return completed

    def _emit_pending_action_rejected(self, pid: str, action: dict[str, Any], request_id: str, error: str) -> None:
        tool_name = str(action.get("action"))
        source = f"tool:{tool_name}"
        try:
            handle = self.runtime.tools.resolve(tool_name, pid=pid)
            source = f"tool:{handle.tool_id}"
        except Exception:
            pass
        self.runtime.events.emit(
            EventType.TOOL_FAILED,
            source=source,
            target=pid,
            payload={
                "error": error,
                "tool_name": tool_name,
                "request_id": request_id,
                "policy_decision": "deny",
                "policy_reason": "human_rejected_per_use_approval",
            },
        )
        self.runtime.audit.record(
            actor=pid,
            action="llm.pending_action_rejected",
            target=tool_name,
            decision={"request_id": request_id, "action": sanitize_for_observability(action), "error": error},
        )

    def _completion_to_action(self, content: str, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
        errors: list[str] = []
        for tool_call in reversed(tool_calls):
            try:
                return tool_call_to_action(tool_call)
            except Exception as exc:
                errors.append(str(exc))
        try:
            return parse_json_action(content)
        except Exception as exc:
            detail = f"; invalid tool calls: {errors}" if errors else ""
            raise ValueError(
                f"no valid tool call or fallback JSON action found: {exc}{detail}; "
                f"content preview: {content[: self.config.llm.content_preview_chars]!r}"
            ) from exc

    async def _complete_valid_action(
        self,
        pid: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_attempts: int | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        attempt_messages = messages
        last_error: Exception | None = None
        selected_max_attempts = max_attempts or self.config.llm.action_repair_attempts
        for attempt in range(selected_max_attempts):
            completion = await self._complete_action_recorded(
                pid=pid,
                messages=attempt_messages,
                tools=tools,
                attempt=attempt + 1,
                max_attempts=selected_max_attempts,
            )
            try:
                action = self.runtime.tools.normalize_model_action(
                    pid,
                    self._completion_to_action(completion.content, completion.tool_calls),
                )
                self._validate_dispatchable_action(pid, action)
                return completion, action
            except ValueError as exc:
                last_error = exc
                self.runtime.audit.record(
                    actor=pid,
                    action="llm.action_repair_requested",
                    target=f"process:{pid}",
                    decision={
                        "attempt": attempt + 1,
                        "error": str(exc),
                        "tool_call_count": len(completion.tool_calls),
                        "tool_calls_preview": self._tool_call_previews(completion.tool_calls),
                        "content_preview": completion.content[: self.config.llm.content_preview_chars],
                    },
                )
                if attempt + 1 >= selected_max_attempts:
                    break
                attempt_messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            "The previous model response could not be dispatched: "
                            f"{exc}. Choose exactly one available OpenAI tool call by its function name. "
                            f"Available tool names: {self.runtime.tools.model_tool_names(pid)}"
                        ),
                    },
                ]
        assert last_error is not None
        raise last_error

    def _validate_dispatchable_action(self, pid: str, action: dict[str, Any]) -> None:
        name = str(action.get("action") or "").strip()
        if not name:
            raise ValueError("selected action has an empty tool name")
        process = self.runtime.process.get(pid)
        if name not in process.tool_table:
            raise ValueError(f"selected action is not in this process tool table: {name}")

    def _tool_call_previews(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        previews: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            raw_args = tool_call.get("arguments")
            if isinstance(raw_args, str):
                raw_bytes = raw_args.encode("utf-8", errors="replace")
                try:
                    observable_args: Any = json.loads(raw_args)
                except ValueError:
                    observable_args = raw_args
            else:
                raw_text = repr(raw_args)
                raw_bytes = raw_text.encode("utf-8", errors="replace")
                observable_args = raw_args
            observation = sanitize_for_observability(
                observable_args,
                preview_chars=self.config.llm.tool_arguments_preview_chars,
            )
            previews.append(
                {
                    "id": tool_call.get("id"),
                    "call_id": tool_call.get("call_id"),
                    "name": tool_call.get("name"),
                    "arguments_type": type(raw_args).__name__,
                    "arguments_preview": observation["preview"],
                    "arguments_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                    "arguments_bytes": len(raw_bytes),
                    "arguments_truncated": observation["truncated"],
                    "arguments_redacted": observation["redacted"],
                }
            )
        return previews

    async def _complete_action_recorded(
        self,
        *,
        pid: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        attempt: int,
        max_attempts: int,
    ) -> Any:
        call_id = new_id("llmcall")
        process = self.runtime.process.get(pid)
        created_at = utc_now()
        profile_id = process.llm_profile_id if process is not None else self.config.llm.default_profile_id
        request_options = {
            "attempt": attempt,
            "max_attempts": max_attempts,
            "purpose": "action_selection",
            "llm_profile_id": profile_id,
        }
        self._preflight_llm_call(pid)
        try:
            resolved = self.runtime.llms.resolve(profile_id)
            client = resolved.client
            request_options.update(
                {
                    "llm_profile_id": resolved.profile_id,
                    "client_class": type(client).__name__,
                    "real_llm_client": isinstance(client, LLMClient),
                }
            )
            completion = await self._complete_action(
                client,
                messages,
                tools,
                temperature=resolved.temperature,
                max_tokens=resolved.max_tokens,
            )
        except Exception as exc:
            self._charge_llm_attempt(pid, source="llm.error", context={"error_type": type(exc).__name__})
            self.runtime.store.insert_llm_call(
                LLMCallRecord(
                    call_id=call_id,
                    pid=pid,
                    image_id=process.image_id if process is not None else None,
                    purpose="action_selection",
                    status="error",
                    **observable_llm_call_fields(
                        messages=messages,
                        tools=tools,
                        response_content="",
                        tool_calls=[],
                        reasoning=None,
                        raw_response=None,
                        config=self.config,
                    ),
                    request_options=request_options,
                    error=str(exc),
                    created_at=created_at,
                    completed_at=utc_now(),
                )
            )
            raise
        self._charge_llm_attempt(pid, source="llm.completion", context={"usage": dict(getattr(completion, "usage", {}) or {})})
        observable_fields = observable_llm_call_fields(
            messages=messages,
            tools=tools,
            response_content=str(getattr(completion, "content", "")),
            tool_calls=list(getattr(completion, "tool_calls", []) or []),
            reasoning=getattr(completion, "reasoning", None),
            raw_response=getattr(completion, "raw", None),
            config=self.config,
        )
        self.runtime.store.insert_llm_call(
            LLMCallRecord(
                call_id=call_id,
                pid=pid,
                image_id=process.image_id if process is not None else None,
                purpose="action_selection",
                status="ok",
                api=getattr(completion, "api", None),
                model=getattr(completion, "model", None),
                request_id=getattr(completion, "request_id", None),
                response_id=getattr(completion, "response_id", None),
                messages=observable_fields["messages"],
                tools=observable_fields["tools"],
                request_options=request_options,
                response_content=observable_fields["response_content"],
                tool_calls=observable_fields["tool_calls"],
                reasoning=observable_fields["reasoning"],
                usage=dict(getattr(completion, "usage", {}) or {}),
                raw_response=observable_fields["raw_response"],
                observability=observable_fields["observability"],
                created_at=created_at,
                completed_at=utc_now(),
            )
        )
        self._charge_llm_completion(pid, completion)
        return completion

    def _preflight_llm_call(self, pid: str) -> None:
        resources = getattr(self.runtime, "resources", None)
        if resources is None:
            return
        resources.preflight(
            pid,
            ResourceUsage(llm_calls=1),
            source="llm.request",
            context={"purpose": "action_selection"},
        )

    def _charge_llm_attempt(self, pid: str, *, source: str, context: dict[str, Any] | None = None) -> None:
        resources = getattr(self.runtime, "resources", None)
        if resources is None:
            return
        resources.charge(
            pid,
            ResourceUsage(llm_calls=1),
            source=source,
            context=context or {},
            allow_overage=True,
            kill_on_exceed=True,
        )

    def _charge_llm_completion(self, pid: str, completion: Any) -> None:
        resources = getattr(self.runtime, "resources", None)
        if resources is None:
            return
        usage = dict(getattr(completion, "usage", {}) or {})
        has_token_limit = resources.has_limit(pid, "max_llm_total_tokens")
        token_keys = {"prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"}
        if has_token_limit and not any(key in usage for key in token_keys):
            resources.charge(
                pid,
                ResourceUsage(),
                source="llm.completion",
                context={"usage_missing": True},
                allow_overage=False,
                kill_on_exceed=False,
            )
            raise ResourceLimitExceeded("LLM token budget is configured, but provider response did not include token usage")
        prompt_tokens = self._usage_int(usage, "prompt_tokens", "input_tokens")
        completion_tokens = self._usage_int(usage, "completion_tokens", "output_tokens")
        total_tokens = self._usage_int(usage, "total_tokens")
        if total_tokens == 0 and (prompt_tokens or completion_tokens):
            total_tokens = prompt_tokens + completion_tokens
        resources.charge(
            pid,
            ResourceUsage(
                llm_prompt_tokens=prompt_tokens,
                llm_completion_tokens=completion_tokens,
                llm_total_tokens=total_tokens,
            ),
            source="llm.completion",
            context={"usage": usage},
            allow_overage=True,
            kill_on_exceed=True,
        )

    def _usage_int(self, usage: dict[str, Any], *keys: str) -> int:
        for key in keys:
            value = usage.get(key)
            if value is None:
                continue
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                continue
        return 0

    async def _complete_action(
        self,
        client: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> Any:
        kwargs = {"temperature": temperature, "max_tokens": max_tokens}
        if hasattr(client, "acomplete_action"):
            result = (
                client.acomplete_action(messages, tools, **kwargs)
                if isinstance(client, LLMClient)
                else client.acomplete_action(messages, tools)
            )
            if inspect.isawaitable(result):
                return await result
            return result
        if isinstance(client, LLMClient):
            return await asyncio.to_thread(client.complete_action, messages, tools, **kwargs)
        return await asyncio.to_thread(client.complete_action, messages, tools)

    def dispatch(
        self,
        pid: str,
        action: dict[str, Any],
        *,
        context_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        name = str(action["action"])
        args = {key: value for key, value in action.items() if key != "action"}
        if notice := self._pre_tool_interrupt_notice(pid, name):
            return notice
        result = self.runtime.tools.call(pid, name, args, context_metadata=context_metadata)
        if result.result_handle is not None:
            self._add_to_view(pid, result.result_handle)
        post_tool_notice = self._notify_normal_messages(pid)
        return {
            "ok": result.ok,
            "tool_id": result.tool_id,
            "result_oid": result.result_handle.oid if result.result_handle else None,
            "payload": result.payload,
            "error": result.error,
            "message_notice": post_tool_notice,
        }

    async def adispatch(
        self,
        pid: str,
        action: dict[str, Any],
        *,
        context_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        name = str(action["action"])
        args = {key: value for key, value in action.items() if key != "action"}
        if notice := self._pre_tool_interrupt_notice(pid, name):
            return notice
        result = await self.runtime.tools.acall(pid, name, args, context_metadata=context_metadata)
        if result.result_handle is not None:
            self._add_to_view(pid, result.result_handle)
        post_tool_notice = self._notify_normal_messages(pid)
        return {
            "ok": result.ok,
            "tool_id": result.tool_id,
            "result_oid": result.result_handle.oid if result.result_handle else None,
            "payload": result.payload,
            "error": result.error,
            "message_notice": post_tool_notice,
        }

    def _notify_interrupt_messages(self, pid: str) -> dict[str, Any] | None:
        return self.runtime.messages.notice(
            pid,
            kind=ProcessMessageKind.INTERRUPT,
            phase="before_llm_tool_selection",
            source="llm.executor",
        )

    def _pre_tool_interrupt_notice(self, pid: str, tool_name: str) -> dict[str, Any] | None:
        if tool_name in {"read_process_messages", "receive_process_messages"}:
            return None
        notice = self.runtime.messages.notice(
            pid,
            kind=ProcessMessageKind.INTERRUPT,
            phase="before_tool_call",
            source="llm.executor",
        )
        if notice is None:
            return None
        return {
            "ok": False,
            "tool_id": None,
            "result_oid": None,
            "payload": {"message_notice": notice},
            "error": "unread interrupt process messages are waiting; call read_process_messages or receive_process_messages first",
            "interrupted_by_message": True,
            "message_notice": notice,
        }

    def _notify_normal_messages(self, pid: str) -> dict[str, Any] | None:
        return self.runtime.messages.notice(
            pid,
            kind=ProcessMessageKind.NORMAL,
            phase="after_tool_call",
            source="llm.executor",
        )

    def _handles_for_oids(self, pid: str, oids: list[str]) -> list[ObjectHandle]:
        return [self._handle_for_oid(pid, oid) for oid in oids]

    def _handle_for_oid(self, pid: str, oid: str) -> ObjectHandle:
        process = self.runtime.process.get(pid)
        if process.memory_view is not None:
            for handle in process.memory_view.roots:
                if handle.oid == oid:
                    return handle
        return self.runtime.memory.handle_for_oid(
            pid,
            oid,
            required_rights={ObjectRight.READ.value},
            optional_rights={ObjectRight.MATERIALIZE.value, ObjectRight.LINK.value, ObjectRight.DIFF.value},
            issued_by="llm.executor",
        )

    def _add_to_view(self, pid: str, handle: ObjectHandle) -> None:
        process = self.runtime.process.get(pid)
        if process.memory_view is None:
            process.memory_view = self.runtime.memory.create_view(pid, [handle], mode=ViewMode.READ_ONLY)
        elif all(existing.oid != handle.oid for existing in process.memory_view.roots):
            process.memory_view.roots.append(handle)
        process.updated_at = utc_now()
        self.runtime.store.update_process(process)

    def _persist_pending_action(
        self,
        pid: str,
        *,
        wait_type: str,
        action: dict[str, Any],
        content_preview: str,
        tool_call_count: int,
        request_id: str | None = None,
        child_pid: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> None:
        # Pending actions are durable process state. They are consumed only after
        # the blocked primitive can be resumed, preserving the original model
        # decision across runtime restarts and human approval latency.
        self.runtime.store.upsert_llm_pending_action(
            pid,
            {
                "wait_type": wait_type,
                "request_id": request_id,
                "child_pid": child_pid,
                "filters": dict(filters or {}),
                "action": dict(action),
                "content_preview": content_preview,
                "tool_call_count": tool_call_count,
                "status": "pending",
            },
        )

    def _clear_pending_action(self, pid: str) -> None:
        self.runtime.store.complete_llm_pending_action(pid)

    def _load_pending_actions(self) -> None:
        for pending in self.runtime.store.list_llm_pending_actions(status="pending"):
            pid = str(pending["pid"])
            wait_type = str(pending["wait_type"])
            action = dict(pending.get("action") or {})
            common = {
                "action": action,
                "content_preview": str(pending.get("content_preview") or ""),
                "tool_call_count": int(pending.get("tool_call_count") or 0),
            }
            if wait_type == "human" and pending.get("request_id"):
                self._pending_human_actions[pid] = {**common, "request_id": str(pending["request_id"])}
            elif wait_type == "child" and pending.get("child_pid"):
                restored = {**common, "child_pid": str(pending["child_pid"])}
                self._pending_wait_actions[pid] = restored
                self._restore_pending_compaction_child_goal({**pending, **restored})
            elif wait_type == "message":
                self._pending_message_actions[pid] = {**common, "filters": dict(pending.get("filters") or {})}

    def _restore_pending_compaction_child_goal(self, pending: dict[str, Any]) -> None:
        try:
            from agent_libos.tools.builtin.context import restore_pending_compaction_child_goal

            restore_pending_compaction_child_goal(self.runtime, pending)
        except Exception as exc:
            self.runtime.audit.record(
                actor="llm.executor",
                action="llm.pending_compaction_child_restore_failed",
                target=f"process:{pending.get('pid')}",
                decision={"error": str(exc), "child_pid": pending.get("child_pid")},
            )
