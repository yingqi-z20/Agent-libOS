from __future__ import annotations

from typing import Any

from agent_libos.llm.client import LLMError
from agent_libos.models.exceptions import ValidationError


def model_prompt_layout(runtime: Any, pid: str) -> str:
    """Keep tool result projections aligned with the caller's current profile."""

    defaults = getattr(getattr(runtime, "config", None), "llm", None)
    configured = str(getattr(defaults, "prompt_layout", "legacy_v1"))
    fallback = "legacy_v1" if configured == "auto" else configured
    registry = getattr(runtime, "llms", None)
    if registry is None:
        # Direct tool consumers and test doubles need not attach an LLM registry.
        return fallback
    process = runtime.process.get(pid)
    profile_id = process.llm_profile_id or defaults.default_profile_id
    try:
        return str(registry.profile_snapshot(profile_id).policy.prompt_layout)
    except (ValidationError, LLMError):
        # The protected LLM call reports invalid profiles. Rendering a tool
        # result must not turn that condition into an unrelated tool failure.
        return fallback
