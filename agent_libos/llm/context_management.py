from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agent_libos.models import is_openai_tool_name
from agent_libos.models.exceptions import ValidationError
from agent_libos.utils.ids import estimate_tokens
from agent_libos.utils.serde import dumps, to_jsonable

CONTEXT_MANAGEMENT_MODES = frozenset({"auto_compact", "prompt", "disabled"})
DEFAULT_CONTEXT_PRESSURE_PROMPT = (
    "Context window pressure is high. Preserve critical state and reduce "
    "context before continuing."
)
_DEFAULT_TOOL_NAME = "compact_process_context"


@dataclass(frozen=True, slots=True)
class ContextManagementPolicy:
    mode: str = "auto_compact"
    threshold_ratio: float = 0.8
    tool_name: str = _DEFAULT_TOOL_NAME
    tool_arguments: dict[str, Any] | None = None
    prompt: str = DEFAULT_CONTEXT_PRESSURE_PROMPT

    def __post_init__(self) -> None:
        if self.tool_arguments is None:
            object.__setattr__(self, "tool_arguments", {})

    @property
    def fingerprint(self) -> str:
        material = {
            "schema_version": 1,
            "mode": self.mode,
            "threshold_ratio": self.threshold_ratio,
            "tool": {
                "name": self.tool_name,
                "arguments": self.tool_arguments,
            },
            "prompt": self.prompt,
        }
        return hashlib.sha256(dumps(to_jsonable(material)).encode("utf-8")).hexdigest()

    def tool_action(self) -> dict[str, Any]:
        return {
            "action": self.tool_name,
            **dict(self.tool_arguments or {}),
        }


@dataclass(frozen=True, slots=True)
class ContextPressureAssessment:
    context_window_tokens: int
    local_input_estimate_tokens: int
    provider_usage_lower_bound_tokens: int
    estimated_input_tokens: int
    reserved_output_tokens: int
    projected_tokens: int
    utilization_ratio: float
    threshold_ratio: float
    triggered: bool
    profile_id: str
    context_generation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "context_window_tokens": self.context_window_tokens,
            "local_input_estimate_tokens": self.local_input_estimate_tokens,
            "provider_usage_lower_bound_tokens": self.provider_usage_lower_bound_tokens,
            "estimated_input_tokens": self.estimated_input_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "projected_tokens": self.projected_tokens,
            "utilization_ratio": self.utilization_ratio,
            "utilization_percent": round(self.utilization_ratio * 100, 2),
            "threshold_ratio": self.threshold_ratio,
            "triggered": self.triggered,
            "profile_id": self.profile_id,
            "context_generation": self.context_generation,
        }


def context_management_policy(planner: Mapping[str, Any] | None) -> ContextManagementPolicy:
    """Return the strict nested Image policy while preserving other planner keys."""

    if not planner or "context_management" not in planner:
        return ContextManagementPolicy()
    raw = planner["context_management"]
    if not isinstance(raw, Mapping):
        raise ValidationError("planner.context_management must be an object")
    if any(not isinstance(key, str) for key in raw):
        raise ValidationError(
            "planner.context_management field names must be strings"
        )
    unknown = sorted(set(raw) - {"mode", "threshold_ratio", "tool", "prompt"})
    if unknown:
        raise ValidationError(
            "unknown planner.context_management fields: " + ", ".join(unknown)
        )

    mode = _context_management_mode(raw)
    threshold = _context_management_threshold(raw)
    tool_name, normalized_arguments = _context_management_tool(raw)
    prompt = _context_management_prompt(raw)
    return ContextManagementPolicy(
        mode=mode,
        threshold_ratio=threshold,
        tool_name=tool_name,
        tool_arguments=normalized_arguments,
        prompt=prompt,
    )


def _context_management_mode(raw: Mapping[str, Any]) -> str:
    mode = raw.get("mode", "auto_compact")
    if not isinstance(mode, str) or mode not in CONTEXT_MANAGEMENT_MODES:
        raise ValidationError(
            "planner.context_management.mode must be one of "
            "auto_compact, prompt, disabled"
        )

    return mode


def _context_management_threshold(raw: Mapping[str, Any]) -> float:
    threshold = raw.get("threshold_ratio", 0.8)
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or float(threshold) <= 0
        or float(threshold) > 1
    ):
        raise ValidationError(
            "planner.context_management.threshold_ratio must be in (0, 1]"
        )
    return float(threshold)


def _context_management_tool(
    raw: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    raw_tool = raw.get("tool", {})
    if raw_tool is None:
        raw_tool = {}
    if not isinstance(raw_tool, Mapping):
        raise ValidationError("planner.context_management.tool must be an object")
    if any(not isinstance(key, str) for key in raw_tool):
        raise ValidationError(
            "planner.context_management.tool field names must be strings"
        )
    unknown_tool = sorted(set(raw_tool) - {"name", "arguments"})
    if unknown_tool:
        raise ValidationError(
            "unknown planner.context_management.tool fields: "
            + ", ".join(unknown_tool)
        )
    tool_name = raw_tool.get("name", _DEFAULT_TOOL_NAME)
    if not isinstance(tool_name, str) or not is_openai_tool_name(tool_name):
        raise ValidationError(
            "planner.context_management.tool.name must match OpenAI tool name syntax"
        )
    arguments = raw_tool.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValidationError(
            "planner.context_management.tool.arguments must be an object"
        )
    if "action" in arguments:
        raise ValidationError(
            "planner.context_management.tool.arguments must not override the reserved action field"
        )
    if any(not isinstance(key, str) for key in arguments):
        raise ValidationError(
            "planner.context_management.tool.arguments keys must be strings"
        )
    try:
        encoded_arguments = json.dumps(
            arguments,
            ensure_ascii=True,
            sort_keys=True,
            allow_nan=False,
        )
        normalized_arguments = json.loads(encoded_arguments)
    except (RecursionError, TypeError, ValueError) as exc:
        raise ValidationError(
            "planner.context_management.tool.arguments must be JSON-serializable"
        ) from exc
    if not isinstance(normalized_arguments, dict):
        raise ValidationError(
            "planner.context_management.tool.arguments must be an object"
        )
    return tool_name, normalized_arguments


def _context_management_prompt(raw: Mapping[str, Any]) -> str:
    prompt = raw.get("prompt", DEFAULT_CONTEXT_PRESSURE_PROMPT)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValidationError(
            "planner.context_management.prompt must be a non-empty string"
        )
    return prompt


def estimate_multilingual_tokens(text: str) -> int:
    """Compatibility alias for the shared Provider-neutral estimator."""

    return estimate_tokens(text)


def estimate_request_input_tokens(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
) -> int:
    message_tokens = sum(
        estimate_tokens(dict(message)) + 8
        for message in messages
    )
    tool_tokens = sum(
        estimate_tokens(dict(tool)) + 12
        for tool in tools
    )
    return max(1, message_tokens + tool_tokens + 16)


def provider_usage_lower_bound(
    call: Any | None,
    *,
    profile_id: str,
    context_generation: str,
    previous_response_id: str | None,
) -> int:
    if call is None:
        return 0
    options = getattr(call, "request_options", {})
    if not isinstance(options, dict):
        return 0
    if options.get("llm_profile_id") != profile_id:
        return 0
    if options.get("llm_context_generation") != context_generation:
        return 0
    if (
        not previous_response_id
        or getattr(call, "api", None) != "responses"
        or str(getattr(call, "response_id", "") or "") != previous_response_id
    ):
        # A previous stateless/chat request is not retained by the Provider,
        # so its input usage says nothing about the current request size.
        return 0
    usage = getattr(call, "usage", {})
    if not isinstance(usage, dict):
        return 0
    lower_bound = _usage_int(usage, "prompt_tokens", "input_tokens")
    lower_bound = max(lower_bound, _usage_int(usage, "total_tokens"))
    return lower_bound


def assess_context_pressure(
    *,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    context_window_tokens: int,
    reserved_output_tokens: int,
    threshold_ratio: float,
    profile_id: str,
    context_generation: str,
    provider_lower_bound_tokens: int = 0,
) -> ContextPressureAssessment:
    local = estimate_request_input_tokens(messages, tools)
    # In a Responses chain the Provider lower bound is retained history;
    # `local` is the new request that will be appended to that history.
    estimated_input = local + max(0, provider_lower_bound_tokens)
    projected = estimated_input + reserved_output_tokens
    ratio = projected / context_window_tokens
    return ContextPressureAssessment(
        context_window_tokens=context_window_tokens,
        local_input_estimate_tokens=local,
        provider_usage_lower_bound_tokens=max(0, provider_lower_bound_tokens),
        estimated_input_tokens=estimated_input,
        reserved_output_tokens=reserved_output_tokens,
        projected_tokens=projected,
        utilization_ratio=ratio,
        threshold_ratio=threshold_ratio,
        triggered=ratio >= threshold_ratio,
        profile_id=profile_id,
        context_generation=context_generation,
    )


def context_pressure_prompt(
    policy: ContextManagementPolicy,
    assessment: ContextPressureAssessment,
) -> str:
    return "\n".join(
        [
            policy.prompt,
            "Context window pressure details:",
            f"- context window: {assessment.context_window_tokens} tokens",
            f"- estimated input: {assessment.estimated_input_tokens} tokens",
            f"- reserved output: {assessment.reserved_output_tokens} tokens",
            f"- projected occupancy: {assessment.projected_tokens} tokens",
            f"- utilization: {assessment.utilization_ratio * 100:.2f}%",
        ]
    )


def _usage_int(usage: Mapping[str, Any], *keys: str) -> int:
    selected = 0
    for key in keys:
        value = usage.get(key)
        if isinstance(value, bool):
            continue
        try:
            candidate = int(value)
        except (TypeError, ValueError):
            continue
        if candidate >= 0:
            selected = max(selected, candidate)
    return selected
