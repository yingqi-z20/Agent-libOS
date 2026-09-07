from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from agent_libos.config import LLMDefaults


def is_official_openai_endpoint(base_url: str | None) -> bool:
    """Classify the already frozen Host endpoint without reading ambient state."""

    if base_url is None:
        return True
    parsed = urlparse(base_url)
    if parsed.scheme and parsed.scheme != "https":
        return False
    host = parsed.hostname if parsed.scheme else urlparse(f"https://{base_url}").hostname
    return host == "api.openai.com"


def is_astra_model(model: str | None) -> bool:
    return isinstance(model, str) and (
        model == "gpt-6-astra" or model.startswith("gpt-6-astra-")
    )


@dataclass(frozen=True)
class ProviderPolicy:
    model: str | None
    api_mode: str
    reasoning_effort: str | None
    reasoning_context: str | None
    responses_replay: bool
    prompt_layout: str
    prompt_cache_mode: str
    prompt_cache_ttl: str | None
    prompt_cache_retention: str | None


def _reasoning_policy(
    defaults: LLMDefaults, *, astra: bool, uses_responses: bool,
    context: str | None, replay: bool | None,
) -> tuple[str | None, bool]:
    selected_context = defaults.reasoning_context if context is None else context
    if selected_context not in {"auto", "current_turn", "all_turns"}:
        raise ValueError("reasoning_context must be auto, current_turn or all_turns")
    if selected_context == "auto":
        selected_context = defaults.openai_reasoning_context if astra else None
    selected_replay = defaults.responses_replay if replay is None else replay
    if selected_replay is None:
        selected_replay = astra and uses_responses
    if type(selected_replay) is not bool:
        raise ValueError("responses_replay must be boolean or null")
    # Chat cannot consume Responses output items, even with an explicit opt-in.
    return selected_context, selected_replay and uses_responses


def _prompt_layout(defaults: LLMDefaults, layout: str | None, *, official: bool) -> str:
    selected = defaults.prompt_layout if layout is None else layout
    if selected == "auto":
        selected = "cache_optimized_v2" if official else "legacy_v1"
    if selected not in {"legacy_v1", "cache_optimized_v2"}:
        raise ValueError("prompt_layout must be auto, legacy_v1 or cache_optimized_v2")
    return selected


def _cache_policy(
    defaults: LLMDefaults, *, official: bool, mode: str | None,
    ttl: str | None, retention: str | None,
) -> tuple[str, str | None, str | None]:
    selected_mode = defaults.prompt_cache_mode if mode is None else mode
    selected_ttl = defaults.prompt_cache_ttl if ttl is None else ttl
    selected_retention = defaults.prompt_cache_retention if retention is None else retention
    if selected_mode == "auto":
        if official and selected_retention is None:
            selected_mode = "implicit"
            selected_ttl = selected_ttl or defaults.openai_prompt_cache_ttl
        else:
            selected_mode = "provider_default"
            selected_ttl = None
    if selected_mode not in {"provider_default", "implicit", "explicit"}:
        raise ValueError("prompt_cache_mode must be auto, provider_default, implicit or explicit")
    return selected_mode, selected_ttl, selected_retention


def resolve_provider_policy(
    *,
    defaults: LLMDefaults,
    base_url: str | None,
    model: str | None,
    api_mode: str | None = None,
    reasoning_effort: str | None = None,
    reasoning_context: str | None = None,
    responses_replay: bool | None = None,
    prompt_layout: str | None = None,
    prompt_cache_mode: str | None = None,
    prompt_cache_ttl: str | None = None,
    prompt_cache_retention: str | None = None,
) -> ProviderPolicy:
    """Resolve endpoint defaults after callers apply profile/environment precedence.

    Explicit legacy modes remain unchanged. ``auto`` is a Host policy token;
    it must never reach a Provider request.
    """

    official = is_official_openai_endpoint(base_url)
    selected_model = model or (defaults.openai_model if official else None)
    selected_api = defaults.api_mode if api_mode is None else api_mode
    uses_responses = selected_api == "responses" or (
        selected_api == "auto" and official
    )
    astra = official and is_astra_model(selected_model)
    context, replay = _reasoning_policy(
        defaults, astra=astra, uses_responses=uses_responses,
        context=reasoning_context, replay=responses_replay,
    )
    mode, ttl, retention = _cache_policy(
        defaults, official=official, mode=prompt_cache_mode,
        ttl=prompt_cache_ttl, retention=prompt_cache_retention,
    )
    return ProviderPolicy(
        model=selected_model,
        api_mode=selected_api,
        reasoning_effort=(
            reasoning_effort
            if reasoning_effort is not None
            else defaults.openai_reasoning_effort if astra else None
        ),
        reasoning_context=context,
        responses_replay=replay,
        prompt_layout=_prompt_layout(defaults, prompt_layout, official=official),
        prompt_cache_mode=mode,
        prompt_cache_ttl=ttl,
        prompt_cache_retention=retention,
    )
