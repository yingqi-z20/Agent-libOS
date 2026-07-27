from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.action_parser import parse_json_action
from agent_libos.llm.tool_protocol import tool_call_to_action
from agent_libos.models import ResourceUsage, ToolCallResult
from agent_libos.models.exceptions import ResourceLimitExceeded
from agent_libos.storage.repositories import ProcessRepository
from agent_libos.tools.base import model_safe_tool_error_message


_AUTO_WAIT_MESSAGE_TOOL = "receive_process_messages"


class LLMActionService:
    """Parse, validate, and dispatch model actions and narrow Host fallbacks."""

    def __init__(
        self,
        *,
        processes: ProcessRepository,
        tools: Any,
        resources: Any | None,
        content_preview_chars: int,
        tool_call_args_hard_limit_bytes: int = (
            DEFAULT_CONFIG.tools.tool_call_args_hard_limit_bytes
        ),
        pre_tool_notice: Callable[[str, str, dict[str, Any]], dict[str, Any] | None],
        post_tool_notice: Callable[[str], dict[str, Any] | None],
        publish_result: Callable[[str, Any], None],
    ) -> None:
        self._processes = processes
        self._tools = tools
        self._resources = resources
        self._content_preview_chars = content_preview_chars
        self._tool_call_args_hard_limit_bytes = tool_call_args_hard_limit_bytes
        self._pre_tool_notice = pre_tool_notice
        self._post_tool_notice = post_tool_notice
        self._publish_result = publish_result

    def completion_to_actions(
        self,
        content: str,
        tool_calls: list[dict[str, Any]],
        *,
        parallel_tool_calls: bool,
        auto_wait_on_empty_tool_calls: bool,
        fallback_json_actions: bool = False,
    ) -> tuple[list[dict[str, Any]], bool]:
        if not parallel_tool_calls:
            if not tool_calls:
                if fallback_json_actions:
                    try:
                        return [parse_json_action(content)], False
                    except Exception:
                        pass
                if auto_wait_on_empty_tool_calls:
                    return [auto_wait_message_action()], True
            return [
                self._single_action(
                    content,
                    tool_calls,
                    fallback_json_actions=fallback_json_actions,
                )
            ], False
        if not tool_calls:
            if fallback_json_actions:
                try:
                    return [parse_json_action(content)], False
                except Exception:
                    pass
            if auto_wait_on_empty_tool_calls:
                return [auto_wait_message_action()], True
            raise ValueError("no native tool calls found in model response")
        return self._parallel_actions(tool_calls), False

    def _single_action(
        self,
        content: str,
        tool_calls: list[dict[str, Any]],
        *,
        fallback_json_actions: bool,
    ) -> dict[str, Any]:
        errors: list[str] = []
        for tool_call in reversed(tool_calls):
            try:
                return self._tool_call_to_action(tool_call)
            except Exception as exc:
                errors.append(str(exc))
        if fallback_json_actions:
            try:
                return parse_json_action(content)
            except Exception as exc:
                errors.append(f"fallback JSON: {exc}")
        detail = f"; validation errors: {errors}" if errors else ""
        preview = content[: self._content_preview_chars]
        raise ValueError(
            f"no valid native tool call found{detail}; content preview: {preview!r}"
        )

    def _parallel_actions(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        errors: list[str] = []
        for index, tool_call in enumerate(tool_calls, start=1):
            try:
                actions.append(self._tool_call_to_action(tool_call))
            except Exception as exc:
                errors.append(f"{index}: {exc}")
        if errors:
            raise ValueError(f"invalid parallel tool calls: {errors}")
        if not actions:
            raise ValueError("parallel tool call response did not include any function calls")
        return actions

    def _tool_call_to_action(
        self,
        tool_call: dict[str, Any],
    ) -> dict[str, Any]:
        return tool_call_to_action(
            tool_call,
            max_argument_bytes=self._tool_call_args_hard_limit_bytes,
        )

    def validate(self, pid: str, action: dict[str, Any]) -> None:
        name = str(action.get("action") or "").strip()
        if not name:
            raise ValueError("selected action has an empty tool name")
        process = self._processes.get_process(pid)
        if process is None:
            raise ValueError(f"selected action process does not exist: {pid}")
        if name not in process.model_tool_table:
            raise ValueError(f"selected action is not in this process model tool projection: {name}")

    def validate_host_auto_wait(self, pid: str, action: dict[str, Any]) -> None:
        """Validate the one host-generated fallback against the full image table."""

        name, args = split_action(action)
        if name != _AUTO_WAIT_MESSAGE_TOOL or args:
            raise ValueError(
                "invalid host-generated empty-tool-call auto-wait action"
            )
        process = self._processes.get_process(pid)
        if process is None:
            raise ValueError(f"host-generated action process does not exist: {pid}")
        expected_tool_id = process.tool_table.get(name)
        if expected_tool_id is None:
            raise ValueError(
                f"host-generated action is not in this process tool table: {name}"
            )
        try:
            handle = self._tools.resolve(name, pid=pid)
        except Exception as exc:
            raise ValueError(
                f"host-generated action cannot resolve its process tool binding: {name}"
            ) from exc
        if handle.tool_id != expected_tool_id:
            raise ValueError(
                f"host-generated action process tool binding changed: {name}"
            )

    def preflight_parallel(self, pid: str, actions: list[dict[str, Any]]) -> None:
        if self._resources is None:
            return
        try:
            self._resources.preflight(
                pid,
                ResourceUsage(tool_calls=len(actions)),
                source="llm.parallel_tool_batch",
                context={
                    "action_count": len(actions),
                    "actions": [str(action.get("action") or "") for action in actions],
                },
            )
        except ResourceLimitExceeded as exc:
            raise ValueError(
                f"parallel tool call batch exceeds remaining tool-call budget: {exc}"
            ) from exc

    def dispatch(
        self,
        pid: str,
        action: dict[str, Any],
        *,
        context_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        name, args = split_action(action)
        if notice := self._pre_tool_notice(pid, name, args):
            return notice
        result = self._tools.call(pid, name, args, context_metadata=context_metadata)
        return self._result(
            pid,
            result,
            publish_result=(
                name not in {"process_exit", "exec_process"}
                or _is_completion_review_result(name, result)
            ),
        )

    async def adispatch(
        self,
        pid: str,
        action: dict[str, Any],
        *,
        context_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        name, args = split_action(action)
        if notice := self._pre_tool_notice(pid, name, args):
            return notice
        result = await self._tools.acall(pid, name, args, context_metadata=context_metadata)
        return self._result(
            pid,
            result,
            publish_result=(
                name not in {"process_exit", "exec_process"}
                or _is_completion_review_result(name, result)
            ),
        )

    async def adispatch_host_auto_wait(
        self,
        pid: str,
        action: dict[str, Any],
        *,
        context_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Dispatch only the executor's verified empty-response wait fallback."""

        self.validate_host_auto_wait(pid, action)
        return await self.adispatch(
            pid,
            action,
            context_metadata=context_metadata,
        )

    def _result(
        self,
        pid: str,
        result: ToolCallResult,
        *,
        publish_result: bool,
    ) -> dict[str, Any]:
        if publish_result and result.result_handle is not None:
            self._publish_result(pid, result.result_handle)
        notice = self._post_tool_notice(pid)
        return {
            "ok": result.ok,
            "tool_id": result.tool_id,
            "result_oid": result.result_handle.oid if result.result_handle else None,
            "payload": result.payload,
            "error": _model_facing_error(result),
            "message_notice": notice,
        }


def auto_wait_message_action() -> dict[str, Any]:
    return {"action": _AUTO_WAIT_MESSAGE_TOOL}


def _model_facing_error(result: ToolCallResult) -> str | None:
    if result.ok or result.error is None:
        return None
    if isinstance(result.payload, dict):
        error = result.payload.get("error")
        if isinstance(error, dict):
            safe_message = error.get("safe_message")
            if isinstance(safe_message, str) and safe_message:
                return safe_message
    return model_safe_tool_error_message(result.error)


def _is_completion_review_result(name: str, result: ToolCallResult) -> bool:
    if name != "process_exit" or not result.ok or not isinstance(result.payload, dict):
        return False
    return result.payload.get("status") == "completion_review_required"


def split_action(action: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    name = str(action["action"])
    return name, {key: value for key, value in action.items() if key != "action"}
