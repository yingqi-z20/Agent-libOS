from __future__ import annotations

import inspect
from typing import Any

from agent_libos.llm.client import (
    LLMClient,
    LLMTransientError,
    llm_provider_failure_error,
)
from agent_libos.ports.blocking_work import run_blocking_once


class LLMProviderService:
    """Isolate provider invocation and sync/async compatibility handling."""

    def __init__(self, blocking_work: Any | None = None) -> None:
        self._blocking_work = blocking_work

    async def _run_sync(self, function: Any, /, *args: Any, **kwargs: Any) -> Any:
        if self._blocking_work is not None:
            return await self._blocking_work.run(function, *args, **kwargs)
        return await run_blocking_once(function, *args, **kwargs)

    async def complete_action(
        self,
        client: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
        previous_response_id: str | None = None,
        parallel_tool_calls: bool,
    ) -> Any:
        try:
            return await self._complete_action_unwrapped(
                client,
                messages,
                tools,
                temperature=temperature,
                max_tokens=max_tokens,
                previous_response_id=previous_response_id,
                parallel_tool_calls=parallel_tool_calls,
            )
        except Exception as exc:
            wrapped = llm_provider_failure_error(
                exc,
                transient=isinstance(exc, LLMTransientError),
            )
            if wrapped is exc:
                raise
            raise wrapped from exc

    async def _complete_action_unwrapped(
        self,
        client: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
        previous_response_id: str | None = None,
        parallel_tool_calls: bool,
    ) -> Any:
        kwargs = {"temperature": temperature, "max_tokens": max_tokens}
        if hasattr(client, "acomplete_action"):
            result = (
                client.acomplete_action(
                    messages,
                    tools,
                    **kwargs,
                    previous_response_id=previous_response_id,
                    parallel_tool_calls=parallel_tool_calls,
                )
                if isinstance(client, LLMClient)
                else client.acomplete_action(messages, tools)
            )
            if inspect.isawaitable(result):
                return await result
            return result
        if isinstance(client, LLMClient):
            return await self._run_sync(
                client.complete_action,
                messages,
                tools,
                **kwargs,
                previous_response_id=previous_response_id,
                parallel_tool_calls=parallel_tool_calls,
            )
        return await self._run_sync(client.complete_action, messages, tools)
