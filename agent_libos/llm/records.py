from __future__ import annotations

from typing import Any

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.evidence.payload_retention import content_free_llm_call_fields
from agent_libos.tools.observability import observation_envelope, sanitize_for_observability


def observable_llm_call_fields(
    *,
    messages: Any,
    tools: Any,
    response_content: Any = "",
    tool_calls: Any = None,
    reasoning: Any = None,
    raw_response: Any = None,
    error: Any = None,
    config: AgentLibOSConfig | None = None,
) -> dict[str, Any]:
    """Build the durable LLM call view.

    LLM prompts, tool arguments, reasoning traces, and raw provider responses
    often contain user data or materialized object content. By default, the
    durable call row stores full values for self-evolution training and
    fine-tuning pipelines. Operators can set ``config.llm.persist_full_io`` to
    ``False`` to retain canonical, content-free summary envelopes instead.
    """

    selected = config or DEFAULT_CONFIG
    preview_chars = selected.llm.call_record_preview_chars
    response_observation = observation_envelope(response_content, preview_chars=preview_chars)
    selected_response_content = "" if response_content is None else str(response_content)
    selected_error = None if error is None else str(error)
    observability = {
        "messages": sanitize_for_observability(messages, preview_chars=preview_chars),
        "tools": sanitize_for_observability(tools, preview_chars=preview_chars),
        "response_content": response_observation,
        "tool_calls": sanitize_for_observability(tool_calls or [], preview_chars=preview_chars),
        "reasoning": sanitize_for_observability(reasoning, preview_chars=preview_chars),
        "raw_response": sanitize_for_observability(raw_response, preview_chars=preview_chars),
        "error": sanitize_for_observability(selected_error, preview_chars=preview_chars),
    }
    if selected.llm.persist_full_io:
        return {
            "messages": messages,
            "tools": tools,
            "response_content": selected_response_content,
            "tool_calls": tool_calls or [],
            "reasoning": reasoning,
            "raw_response": raw_response,
            "observability": observability,
            "error": selected_error,
        }
    return content_free_llm_call_fields(
        messages=messages,
        tools=tools,
        response_content=selected_response_content,
        tool_calls=tool_calls or [],
        reasoning=reasoning,
        raw_response=raw_response,
        error=selected_error,
        source_observability=observability,
    )
