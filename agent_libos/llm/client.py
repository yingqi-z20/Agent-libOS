from __future__ import annotations

import asyncio
import contextvars
import email.utils
import hashlib
import inspect
import json
import os
import random
import threading
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig, LLMDefaults
from agent_libos.utils.openai_schema import (
    normalize_openai_chat_tool_schema,
    normalize_openai_structured_output_schema,
    openai_responses_tool_schema,
)
from agent_libos.models.exceptions import LibOSError
from agent_libos.ports.blocking_work import run_blocking_once
from agent_libos.llm.provider_trace import (
    ProviderAttemptKind,
    ProviderTraceBuilder,
    attach_provider_trace,
    project_provider_raw_response,
)
from agent_libos.utils.public_errors import (
    internal_error_observation,
    internal_exception_observation,
    public_error_envelope,
)
from agent_libos.utils.serde import to_jsonable

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_API_MODES = {"auto", "responses", "chat"}
_PROMPT_CACHE_MODES = {"provider_default", "implicit", "explicit"}
_CACHE_STABLE_MESSAGE_KEY = "_agent_libos_cache_stable"
_CACHE_STABLE_PREFIX_CHARS_KEY = "_agent_libos_cache_stable_prefix_chars"
_CACHE_STABLE_PROJECTION_KEY = "_agent_libos_cache_stable_projection"
_ACTIVE_PROVIDER_TRACE: contextvars.ContextVar[ProviderTraceBuilder | None] = (
    contextvars.ContextVar("agent_libos_provider_trace", default=None)
)
_ACTIVE_PROVIDER_ATTEMPT_KIND: contextvars.ContextVar[ProviderAttemptKind] = (
    contextvars.ContextVar("agent_libos_provider_attempt_kind", default="initial")
)

# These are inbound trust-boundary limits, not generation preferences. They
# cap provider-authored material before it is joined or copied into durable
# runtime state. Keep them independent of outbound max-token configuration.
LLM_RESPONSE_CONTENT_MAX_CHARS = 262_144
LLM_RESPONSE_TEXT_MAX_BYTES = 1_048_576
LLM_RESPONSE_TOOL_CALL_MAX_COUNT = 256
LLM_RESPONSE_TOOL_ARGUMENT_MAX_CHARS = 262_144
LLM_RESPONSE_TOOL_ARGUMENT_TOTAL_MAX_CHARS = 1_048_576
LLM_RESPONSE_TOOL_ARGUMENT_TOTAL_MAX_BYTES = 1_048_576
LLM_RESPONSE_OUTPUT_MAX_ITEMS = 2_048
LLM_RESPONSE_CONTENT_MAX_PARTS = 2_048


class LLMError(LibOSError):
    pass


class LLMTransientError(LLMError):
    """Provider failure that is safe to retry in a later process quantum."""


_PROVIDER_FAILURE_MARKER = object()
_PROVIDER_FAILURE_MARKER_ATTR = "_agent_libos_provider_failure_marker"
_PROVIDER_FAILURE_OBSERVATION_ATTR = (
    "_agent_libos_provider_failure_observation"
)


@dataclass
class LLMCompletion:
    content: str
    tool_calls: list[dict[str, Any]]
    raw: Any | None = None
    api: str | None = None
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    reasoning: Any | None = None
    fallback_json_action_used: bool = False
    # Safe, model-facing provider option telemetry. Cache keys and safety
    # identifiers are intentionally represented as booleans rather than copied
    # into durable call records by downstream consumers.
    provider_request_options: dict[str, Any] = field(default_factory=dict)
    compatibility_removed_options: list[str] = field(default_factory=list)
    provider_trace: dict[str, Any] | None = None
    _provider_attempt_sequence: int | None = field(default=None, repr=False)


@dataclass(frozen=True)
class _ProviderCallResult:
    response: Any
    request: dict[str, Any]
    compatibility_removed_options: tuple[str, ...] = ()
    attempt_sequence: int | None = None


@dataclass
class LLMClient:
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout: float | None = None
    max_retries: int | None = None
    api_mode: Literal["auto", "responses", "chat"] | None = None
    store: bool | None = None
    reasoning_effort: str | None = None
    verbosity: Literal["low", "medium", "high"] | None = None
    safety_identifier: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: Literal["in_memory", "24h"] | None = None
    prompt_cache_mode: Literal["provider_default", "implicit", "explicit"] | None = None
    prompt_cache_ttl: Literal["30m"] | None = None
    responses_previous_response_id: bool | None = None
    parallel_tool_calls: bool | None = None
    fallback_json_actions: bool | None = None
    enable_thinking: bool | None = None
    organization: str | None = None
    project: str | None = None
    inherit_ambient_openai_sdk_config: bool = True
    allow_custom_base_url: bool = False
    defaults: LLMDefaults = field(default_factory=lambda: DEFAULT_CONFIG.llm, repr=False)
    _client: Any | None = field(default=None, init=False, repr=False)
    _async_client: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.timeout = self.defaults.timeout_s if self.timeout is None else self.timeout
        self.max_retries = self.defaults.max_retries if self.max_retries is None else self.max_retries
        self.api_mode = self.defaults.api_mode if self.api_mode is None else self.api_mode
        self.store = self.defaults.store if self.store is None else self.store
        self.safety_identifier = self.defaults.safety_identifier if self.safety_identifier is None else self.safety_identifier
        self.prompt_cache_key = self.defaults.prompt_cache_key if self.prompt_cache_key is None else self.prompt_cache_key
        self.prompt_cache_retention = _normalize_prompt_cache_retention(
            self.defaults.prompt_cache_retention
            if self.prompt_cache_retention is None
            else self.prompt_cache_retention,
            label="prompt_cache_retention",
        )
        self.prompt_cache_mode = _normalize_prompt_cache_mode(
            self.defaults.prompt_cache_mode
            if self.prompt_cache_mode is None
            else self.prompt_cache_mode,
            label="prompt_cache_mode",
        )
        self.prompt_cache_ttl = _normalize_prompt_cache_ttl(
            self.defaults.prompt_cache_ttl
            if self.prompt_cache_ttl is None
            else self.prompt_cache_ttl,
            label="prompt_cache_ttl",
        )
        _validate_prompt_cache_options(
            mode=self.prompt_cache_mode,
            key=self.prompt_cache_key,
            retention=self.prompt_cache_retention,
            ttl=self.prompt_cache_ttl,
        )
        self.responses_previous_response_id = (
            self.defaults.responses_previous_response_id
            if self.responses_previous_response_id is None
            else self.responses_previous_response_id
        )
        self.parallel_tool_calls = (
            self.defaults.parallel_tool_calls if self.parallel_tool_calls is None else self.parallel_tool_calls
        )
        self.fallback_json_actions = (
            self.defaults.fallback_json_actions
            if self.fallback_json_actions is None
            else self.fallback_json_actions
        )
        if self.inherit_ambient_openai_sdk_config and self.base_url is None:
            # Freeze the SDK's ambient endpoint before policy validation.  If
            # this remains unset, the OpenAI SDK re-reads OPENAI_BASE_URL at
            # lazy client construction and can dispatch to an endpoint that
            # Agent libOS never authorized or included in provider identity.
            self.base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        self._validate_base_url_policy()

    @classmethod
    def from_env(
        cls,
        env_path: str | Path | None = None,
        *,
        config: AgentLibOSConfig | LLMDefaults | None = None,
        allow_custom_base_url: bool | None = None,
    ) -> "LLMClient":
        defaults = _llm_defaults(config)
        env = dict(os.environ)
        if env_path is not None:
            for key, value in read_dotenv(env_path).items():
                env.setdefault(key, value)
        api_mode = env.get("OPENAI_API_MODE", defaults.api_mode).strip().lower()
        if api_mode not in _API_MODES:
            raise LLMError(f"OPENAI_API_MODE must be one of {sorted(_API_MODES)}, got {api_mode!r}")
        selected_allow_custom_base_url = (
            _bool_env_from(env, "AGENT_LIBOS_ALLOW_CUSTOM_LLM_BASE_URL", default=False)
            if allow_custom_base_url is None
            else allow_custom_base_url
        )
        return cls(
            base_url=env.get("OPENAI_BASE_URL"),
            model=env.get("OPENAI_LANGUAGE_MODEL") or env.get("OPENAI_MODEL"),
            api_key=env.get("OPENAI_API_KEY"),
            api_key_env="OPENAI_API_KEY",
            timeout=_float_env_from(env, "OPENAI_TIMEOUT", default=defaults.timeout_s),
            max_retries=_int_env_from(env, "OPENAI_MAX_RETRIES", default=defaults.max_retries),
            api_mode=api_mode,  # type: ignore[arg-type]
            store=_bool_env_from(env, "OPENAI_STORE", default=defaults.store),
            reasoning_effort=_optional_env_from(env, "OPENAI_REASONING_EFFORT"),
            verbosity=_verbosity_env_from(env, "OPENAI_VERBOSITY"),
            safety_identifier=_optional_env_from(env, "OPENAI_SAFETY_IDENTIFIER") or defaults.safety_identifier,
            prompt_cache_key=_optional_env_from(env, "OPENAI_PROMPT_CACHE_KEY") or defaults.prompt_cache_key,
            prompt_cache_retention=(
                _prompt_cache_retention_env_from(env, "OPENAI_PROMPT_CACHE_RETENTION")
                or defaults.prompt_cache_retention
            ),
            prompt_cache_mode=(
                _prompt_cache_mode_env_from(env, "OPENAI_PROMPT_CACHE_MODE")
                or defaults.prompt_cache_mode
            ),
            prompt_cache_ttl=(
                _prompt_cache_ttl_env_from(env, "OPENAI_PROMPT_CACHE_TTL")
                or defaults.prompt_cache_ttl
            ),
            responses_previous_response_id=_bool_env_from(
                env,
                "OPENAI_RESPONSES_PREVIOUS_RESPONSE_ID",
                default=defaults.responses_previous_response_id,
            ),
            parallel_tool_calls=_bool_env_from(
                env,
                "OPENAI_PARALLEL_TOOL_CALLS",
                default=defaults.parallel_tool_calls,
            ),
            fallback_json_actions=_bool_env_from(
                env,
                "OPENAI_FALLBACK_JSON_ACTIONS",
                default=defaults.fallback_json_actions,
            ),
            enable_thinking=(
                _bool_env_from(env, "OPENAI_ENABLE_THINKING", default=False)
                if "OPENAI_ENABLE_THINKING" in env
                else None
            ),
            organization=(
                _optional_env_from(env, "OPENAI_ORGANIZATION")
                or _optional_env_from(env, "OPENAI_ORG_ID")
            ),
            project=(
                _optional_env_from(env, "OPENAI_PROJECT")
                or _optional_env_from(env, "OPENAI_PROJECT_ID")
            ),
            inherit_ambient_openai_sdk_config=False,
            allow_custom_base_url=selected_allow_custom_base_url,
            defaults=defaults,
        )

    def close(self) -> None:
        for attr in ("_client", "_async_client"):
            client = getattr(self, attr)
            if client is None:
                continue
            close = getattr(client, "close", None) or getattr(client, "aclose", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    _run_close_sync(result)
            setattr(self, attr, None)

    shutdown = close

    async def aclose(self) -> None:
        for attr in ("_client", "_async_client"):
            client = getattr(self, attr)
            if client is None:
                continue
            close = getattr(client, "close", None) or getattr(client, "aclose", None)
            if callable(close):
                if inspect.iscoroutinefunction(close) or inspect.iscoroutinefunction(
                    getattr(close, "__call__", None)
                ):
                    result = close()
                else:
                    result = await run_blocking_once(close)
                if inspect.isawaitable(result):
                    await result
            setattr(self, attr, None)

    ashutdown = aclose

    def complete(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = True,
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
    ) -> str:
        return self.complete_with_metadata(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            json_schema=json_schema,
            schema_name=schema_name,
        ).content

    def complete_with_metadata(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = True,
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
    ) -> LLMCompletion:
        return _run_sync(
            self.acomplete_with_metadata(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
                json_schema=json_schema,
                schema_name=schema_name,
            )
        )

    async def acomplete(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = True,
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
    ) -> str:
        return (
            await self.acomplete_with_metadata(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
                json_schema=json_schema,
                schema_name=schema_name,
            )
        ).content

    async def acomplete_with_metadata(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = True,
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
    ) -> LLMCompletion:
        trace = ProviderTraceBuilder()
        trace_token = _ACTIVE_PROVIDER_TRACE.set(trace)
        kind_token = _ACTIVE_PROVIDER_ATTEMPT_KIND.set("initial")
        try:
            selected_messages = (
                self._messages_with_json_instruction(messages)
                if json_mode and json_schema is None
                else messages
            )
            completion = await self._complete_without_tools(
                messages=selected_messages,
                temperature=self._temperature(temperature),
                max_tokens=self._max_tokens(max_tokens),
                json_mode=json_mode,
                json_schema=json_schema,
                schema_name=schema_name,
            )
            if not completion.content:
                error = llm_provider_failure_error(
                    "empty content",
                    diagnostic_type="ProviderEmptyResponse",
                )
                _reject_active_provider_sequence(
                    completion._provider_attempt_sequence,
                    error,
                )
                raise error
            trace.mark_selected(completion._provider_attempt_sequence)
            completion.provider_trace = trace.to_dict()
            return completion
        except BaseException as exc:
            attach_provider_trace(exc, trace.to_dict())
            raise
        finally:
            _ACTIVE_PROVIDER_ATTEMPT_KIND.reset(kind_token)
            _ACTIVE_PROVIDER_TRACE.reset(trace_token)

    def complete_action(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        previous_response_id: str | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> LLMCompletion:
        return _run_sync(
            self.acomplete_action(
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                previous_response_id=previous_response_id,
                parallel_tool_calls=parallel_tool_calls,
            )
        )

    async def acomplete_action(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        previous_response_id: str | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> LLMCompletion:
        trace = ProviderTraceBuilder()
        trace_token = _ACTIVE_PROVIDER_TRACE.set(trace)
        kind_token = _ACTIVE_PROVIDER_ATTEMPT_KIND.set("initial")
        try:
            completion = await self._acomplete_action_untraced(
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                previous_response_id=previous_response_id,
                parallel_tool_calls=parallel_tool_calls,
            )
            trace.mark_selected(completion._provider_attempt_sequence)
            completion.provider_trace = trace.to_dict()
            return completion
        except BaseException as exc:
            attach_provider_trace(exc, trace.to_dict())
            raise
        finally:
            _ACTIVE_PROVIDER_ATTEMPT_KIND.reset(kind_token)
            _ACTIVE_PROVIDER_TRACE.reset(trace_token)

    async def _acomplete_action_untraced(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        previous_response_id: str | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> LLMCompletion:
        selected_temperature = self._temperature(temperature)
        selected_max_tokens = self._max_tokens(max_tokens)
        selected_parallel_tool_calls = self._parallel_tool_calls(parallel_tool_calls)
        if self._use_responses_api():
            try:
                return await self._responses_complete_action(
                    messages,
                    tools,
                    selected_temperature,
                    selected_max_tokens,
                    previous_response_id=previous_response_id,
                    parallel_tool_calls=selected_parallel_tool_calls,
                )
            except LLMError as exc:
                cause = exc.__cause__ or exc
                if self.api_mode == "auto" and self._should_fallback_to_chat(cause):
                    pass
                elif self.fallback_json_actions and self._is_tool_protocol_rejection(
                    exc.__cause__ or exc
                ):
                    with _provider_attempt_kind("json_action_fallback"):
                        return await self._complete_json_action_fallback(
                            messages,
                            selected_temperature,
                            selected_max_tokens,
                        )
                else:
                    raise
            with _provider_attempt_kind("responses_to_chat"):
                return await self._chat_complete_action(
                    messages,
                    tools,
                    selected_temperature,
                    selected_max_tokens,
                    parallel_tool_calls=selected_parallel_tool_calls,
                )
        return await self._chat_complete_action(
            messages,
            tools,
            selected_temperature,
            selected_max_tokens,
            parallel_tool_calls=selected_parallel_tool_calls,
        )

    async def _complete_without_tools(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        json_schema: dict[str, Any] | None,
        schema_name: str,
    ) -> LLMCompletion:
        if self._use_responses_api():
            try:
                return await self._responses_complete(messages, temperature, max_tokens, json_mode, json_schema, schema_name)
            except LLMError as exc:
                if self.api_mode != "auto" or not self._should_fallback_to_chat(exc.__cause__ or exc):
                    raise
            with _provider_attempt_kind("responses_to_chat"):
                return await self._chat_complete(
                    messages,
                    temperature,
                    max_tokens,
                    json_mode,
                    json_schema,
                    schema_name,
                )
        return await self._chat_complete(messages, temperature, max_tokens, json_mode, json_schema, schema_name)

    async def _responses_complete(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        json_schema: dict[str, Any] | None,
        schema_name: str,
    ) -> LLMCompletion:
        payload = self._responses_payload(messages, temperature=temperature, max_tokens=max_tokens)
        if json_schema is not None:
            payload["text"] = self._responses_text_config_for_schema(json_schema, schema_name)
        elif json_mode:
            payload["text"] = self._text_config(json_mode=True)
        provider_call = await self._create_response(payload)
        try:
            return self._completion_from_response(provider_call)
        except Exception as exc:
            _reject_active_provider_attempt(provider_call, exc)
            raise

    async def _responses_complete_action(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        *,
        previous_response_id: str | None = None,
        parallel_tool_calls: bool,
    ) -> LLMCompletion:
        payload = self._responses_payload(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            previous_response_id=previous_response_id,
        )
        payload.update(
            {
                "tools": _responses_tools_from_chat_tools(tools),
                "tool_choice": "auto",
                "parallel_tool_calls": parallel_tool_calls,
            }
        )
        provider_call = await self._create_response(payload)
        try:
            return self._completion_from_response(provider_call)
        except Exception as exc:
            _reject_active_provider_attempt(provider_call, exc)
            raise

    async def _chat_complete(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        json_schema: dict[str, Any] | None,
        schema_name: str,
    ) -> LLMCompletion:
        payload = self._chat_payload(messages=messages, temperature=temperature, max_tokens=max_tokens)
        if json_schema is not None:
            payload["response_format"] = self._chat_response_format_for_schema(json_schema, schema_name)
        elif json_mode:
            payload["response_format"] = {"type": "json_object"}
        provider_call = await self._create_chat_completion(payload)
        try:
            result = self._completion_from_chat(provider_call)
        except Exception as exc:
            _reject_active_provider_attempt(provider_call, exc)
            raise
        if self._needs_non_thinking_retry(result):
            _reject_active_provider_attempt(
                provider_call,
                RuntimeError("Provider completion required non-thinking retry"),
            )
            previous_removed = provider_call.compatibility_removed_options
            retry_payload = self._with_enable_thinking(provider_call.request, enabled=False)
            with _provider_attempt_kind("non_thinking_retry"):
                retry_call = await self._create_chat_completion(retry_payload)
            try:
                result = self._completion_from_chat(
                    retry_call,
                    additional_removed=previous_removed,
                )
            except Exception as exc:
                _reject_active_provider_attempt(retry_call, exc)
                raise
            provider_call = retry_call
        if not result.content:
            finish_reason = _first_choice_attr(provider_call.response, "finish_reason")
            error = llm_provider_failure_error(
                finish_reason,
                diagnostic_type="ProviderEmptyResponse",
            )
            _reject_active_provider_attempt(provider_call, error)
            raise error
        return result

    async def _chat_complete_action(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        *,
        parallel_tool_calls: bool,
    ) -> LLMCompletion:
        payload = self._chat_payload(messages=messages, temperature=temperature, max_tokens=max_tokens)
        payload.update({"tools": _chat_tools(tools), "tool_choice": "auto", "parallel_tool_calls": parallel_tool_calls})
        try:
            provider_call = await self._create_chat_completion(payload)
        except LLMError as exc:
            if self.fallback_json_actions and self._is_tool_protocol_rejection(
                exc.__cause__ or exc
            ):
                with _provider_attempt_kind("json_action_fallback"):
                    return await self._complete_json_action_fallback(
                        messages,
                        temperature,
                        max_tokens,
                    )
            raise
        try:
            result = self._completion_from_chat(provider_call)
        except Exception as exc:
            _reject_active_provider_attempt(provider_call, exc)
            raise
        if self._needs_non_thinking_retry(result):
            _reject_active_provider_attempt(
                provider_call,
                RuntimeError("Provider completion required non-thinking retry"),
            )
            with _provider_attempt_kind("non_thinking_retry"):
                retry_call = await self._create_chat_completion(
                    self._with_enable_thinking(provider_call.request, enabled=False)
                )
            try:
                result = self._completion_from_chat(
                    retry_call,
                    additional_removed=provider_call.compatibility_removed_options,
                )
            except Exception as exc:
                _reject_active_provider_attempt(retry_call, exc)
                raise
            provider_call = retry_call
        finish_reason = _first_choice_attr(provider_call.response, "finish_reason")
        if finish_reason in {"length", "content_filter"}:
            error = llm_provider_failure_error(
                finish_reason,
                diagnostic_type="ProviderFinishReason",
            )
            _reject_active_provider_attempt(provider_call, error)
            raise error
        return result

    async def _complete_json_action_fallback(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> LLMCompletion:
        result = await self._complete_without_tools(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=False,
            json_schema=None,
            schema_name="response",
        )
        if not result.content:
            error = llm_provider_failure_error(
                "empty JSON action fallback content",
                diagnostic_type="ProviderEmptyResponse",
            )
            _reject_active_provider_sequence(result._provider_attempt_sequence, error)
            raise error
        result.fallback_json_action_used = True
        return result

    def _client_or_raise(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError("The OpenAI Python SDK is not installed. Install it with `pip install openai`.") from exc
        self._client = self._normalize_openai_sdk_client(OpenAI(**self._client_kwargs()))
        return self._client

    def _async_client_or_raise(self) -> Any:
        if self._async_client is not None:
            return self._async_client
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise LLMError("The OpenAI Python SDK is not installed. Install it with `pip install openai`.") from exc
        # Scheduler quanta may run on different short-lived event loops.  An
        # AsyncOpenAI/httpx connection pool is bound to the loop that first
        # opened its keep-alive connections, so caching it on this long-lived
        # profile client makes the next quantum reuse a transport owned by a
        # closed (or different worker) loop.  Keep explicitly injected clients
        # for tests/hosts, but scope real SDK clients to one request.
        return self._normalize_openai_sdk_client(AsyncOpenAI(**self._client_kwargs()))

    @staticmethod
    def _normalize_openai_sdk_client(client: Any) -> Any:
        """Remove unsupported SDK-owned ambient state before the first request."""
        # The SDK reads OPENAI_CUSTOM_HEADERS inside its constructor even when
        # endpoint and account selectors are supplied explicitly. Agent libOS
        # does not expose or identity-bind per-profile custom headers, so none
        # are authorized. Clear them before the client can issue a request.
        # Do not consult the environment again here: another thread can remove
        # it after the SDK captured the headers but before construction returns.
        custom_headers = getattr(client, "_custom_headers", None)
        if not isinstance(custom_headers, dict):
            raise LLMError(
                "the installed OpenAI SDK cannot safely isolate custom headers"
            )
        custom_headers.clear()

        # These fields are also SDK ambient fallbacks, not Agent libOS profile
        # settings. They are unused by provider calls and must not become hidden
        # state outside the resolved profile snapshot.
        if hasattr(client, "admin_api_key"):
            client.admin_api_key = None
        if hasattr(client, "webhook_secret"):
            client.webhook_secret = None
        if getattr(client, "max_retries", None) != 0:
            raise LLMError("the OpenAI SDK client must disable hidden retries")
        transport = getattr(client, "_client", None)
        if transport is None or not hasattr(transport, "follow_redirects"):
            raise LLMError(
                "the installed OpenAI SDK cannot safely disable redirects"
            )
        transport.follow_redirects = False
        if transport.follow_redirects is not False:
            raise LLMError(
                "the installed OpenAI SDK cannot safely disable redirects"
            )
        return client

    @asynccontextmanager
    async def _async_client_scope(self) -> Any:
        client = self._async_client_or_raise()
        owned = self._async_client is None
        try:
            yield client
        finally:
            if owned:
                close = getattr(client, "aclose", None) or getattr(client, "close", None)
                if callable(close):
                    result = close()
                    if inspect.isawaitable(result):
                        await result

    def _client_kwargs(self) -> dict[str, Any]:
        self._validate_base_url_policy()
        if not self.model:
            raise LLMError("OPENAI_LANGUAGE_MODEL or OPENAI_MODEL is not configured")
        # Profile-registry and ``from_env`` callers pass a frozen credential
        # value together with ``inherit_ambient_openai_sdk_config=False``.  In
        # that mode a missing/empty value must remain missing: consulting the
        # environment here would let a credential introduced after profile
        # authorization change the provider account at lazy SDK construction.
        api_key = self.api_key
        if not api_key and self.inherit_ambient_openai_sdk_config:
            api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise LLMError(f"{self.api_key_env} is not configured")

        kwargs: dict[str, Any] = {
            "timeout": self.timeout,
            # Runtime policy records every wire attempt, so SDK-owned hidden
            # retries must remain disabled. ``self.max_retries`` is consumed by
            # the explicit loop around each create call instead.
            "max_retries": 0,
        }
        if api_key:
            kwargs["api_key"] = api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        elif not self.inherit_ambient_openai_sdk_config:
            # The OpenAI SDK otherwise re-reads OPENAI_BASE_URL even when the
            # resolved Agent libOS profile intentionally selected the default
            # OpenAI endpoint.
            kwargs["base_url"] = "https://api.openai.com/v1"
        if self.organization is not None:
            kwargs["organization"] = self.organization
        elif not self.inherit_ambient_openai_sdk_config:
            # Empty values are explicit to the SDK and therefore suppress its
            # ambient OPENAI_ORG_ID/OPENAI_PROJECT_ID fallbacks. Any generated
            # account-routing headers therefore contain no ambient selector.
            kwargs["organization"] = ""
        if self.project is not None:
            kwargs["project"] = self.project
        elif not self.inherit_ambient_openai_sdk_config:
            kwargs["project"] = ""
        return kwargs

    def _responses_payload(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        previous_response_id: str | None = None,
    ) -> dict[str, Any]:
        if not self.model:
            raise LLMError("OPENAI_LANGUAGE_MODEL or OPENAI_MODEL is not configured")
        will_use_previous_response_id = bool(
            previous_response_id
            and self.store
            and self._use_openai_request_options()
            and not _messages_have_unrepresentable_tool_output(messages)
        )
        provider_messages = _messages_for_provider(
            messages,
            api="responses",
            add_cache_breakpoint=self.prompt_cache_mode == "explicit",
        )
        instructions, input_items = _messages_to_responses_parts(
            provider_messages,
            native_tool_outputs=will_use_previous_response_id,
            tool_output_max_chars=self.defaults.tool_output_prompt_max_chars,
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "max_output_tokens": max_tokens,
            "store": self.store,
            "truncation": "auto",
        }
        if instructions:
            payload["instructions"] = instructions
        if self.prompt_cache_mode == "implicit":
            payload[_CACHE_STABLE_PROJECTION_KEY] = (
                _host_stable_message_projection(messages)
            )
        if temperature is not None:
            payload["temperature"] = temperature
        if will_use_previous_response_id:
            payload["previous_response_id"] = previous_response_id
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        text_config = self._text_config(json_mode=False)
        if text_config:
            payload["text"] = text_config
        self._add_provider_request_options(payload)
        extra_body = self._extra_body()
        if extra_body:
            payload["extra_body"] = extra_body
        return payload

    def _chat_payload(self, messages: list[dict[str, Any]], temperature: float, max_tokens: int) -> dict[str, Any]:
        if not self.model:
            raise LLMError("OPENAI_LANGUAGE_MODEL or OPENAI_MODEL is not configured")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _messages_for_provider(
                messages,
                api="chat",
                add_cache_breakpoint=self.prompt_cache_mode == "explicit",
            ),
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if self.store:
            payload["store"] = True
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.verbosity:
            payload["verbosity"] = self.verbosity
        if self.prompt_cache_mode == "implicit":
            payload[_CACHE_STABLE_PROJECTION_KEY] = (
                _host_stable_message_projection(messages)
            )
        self._add_provider_request_options(payload)
        extra_body = self._extra_body()
        if extra_body:
            payload["extra_body"] = extra_body
        return payload

    async def _create_response(self, payload: dict[str, Any]) -> _ProviderCallResult:
        self._finalize_prompt_cache_request(payload)
        async with self._async_client_scope() as client:
            return await self._call_with_compatibility(client.responses.create, payload, api="responses")

    async def _create_chat_completion(self, payload: dict[str, Any]) -> _ProviderCallResult:
        self._finalize_prompt_cache_request(payload)
        async with self._async_client_scope() as client:
            return await self._call_with_compatibility(client.chat.completions.create, payload, api="chat")

    async def _call_with_compatibility(
        self,
        create: Any,
        payload: dict[str, Any],
        api: str,
    ) -> _ProviderCallResult:
        request = dict(payload)
        last_error: Exception | None = None
        removed_options: set[str] = set()
        base_kind = _ACTIVE_PROVIDER_ATTEMPT_KIND.get()
        for compatibility_attempt in range(
            self.defaults.compatibility_retry_attempts
        ):
            try:
                response, sequence = await self._call_with_transport_retries(
                    create,
                    request,
                    api=api,
                    kind=(
                        base_kind
                        if compatibility_attempt == 0
                        else "compatibility_retry"
                    ),
                )
                return _ProviderCallResult(
                    response=response,
                    request=dict(request),
                    compatibility_removed_options=tuple(sorted(removed_options)),
                    attempt_sequence=sequence,
                )
            except Exception as exc:
                if not _is_openai_sdk_error(exc):
                    raise
                last_error = exc
                retry = self._compatibility_retry_payload(request, exc, api=api)
                if retry is None:
                    raise _openai_sdk_request_error(exc) from exc
                removed_options.update(
                    key for key in request if key not in retry
                )
                if (
                    _prompt_cache_breakpoint_count(retry)
                    < _prompt_cache_breakpoint_count(request)
                ):
                    removed_options.add("prompt_cache_breakpoint")
                request = retry
        assert last_error is not None
        raise _openai_sdk_request_error(last_error) from last_error

    async def _call_with_transport_retries(
        self,
        create: Any,
        request: dict[str, Any],
        *,
        api: str,
        kind: ProviderAttemptKind,
    ) -> tuple[Any, int | None]:
        max_retries = max(0, int(self.max_retries or 0))
        for retry_index in range(max_retries + 1):
            trace = _ACTIVE_PROVIDER_TRACE.get()
            sequence = (
                trace.start_attempt(
                    api="responses" if api == "responses" else "chat",
                    kind=kind if retry_index == 0 else "transport_retry",
                )
                if trace is not None
                else None
            )
            try:
                response = await create(**request)
            except Exception as exc:
                if trace is not None and sequence is not None:
                    trace.finish_error(sequence, exc)
                if (
                    not _is_openai_sdk_error(exc)
                    or not _should_retry_openai_sdk_error(exc)
                    or retry_index >= max_retries
                ):
                    raise
                await asyncio.sleep(_openai_retry_delay(exc, retry_index))
                continue
            if trace is not None and sequence is not None:
                trace.finish_response(sequence, response)
            return response, sequence
        raise AssertionError("unreachable Provider retry loop")

    def _compatibility_retry_payload(
        self,
        payload: dict[str, Any],
        exc: Exception,
        api: str,
    ) -> dict[str, Any] | None:
        message = str(exc).lower()
        retry = dict(payload)

        if "enable_thinking" in message and "extra_body" in retry:
            retry.pop("extra_body", None)
            return retry
        if "max_completion_tokens" in message and "max_completion_tokens" in retry:
            retry["max_tokens"] = retry.pop("max_completion_tokens")
            return retry
        if "max_tokens" in message and "max_tokens" in retry:
            retry["max_completion_tokens"] = retry.pop("max_tokens")
            return retry
        if "max_output_tokens" in message and "max_output_tokens" in retry and api == "responses":
            return None
        if "strict" in message and isinstance(retry.get("tools"), list):
            retry["tools"] = _tools_without_strict(retry["tools"])
            return retry
        cache_retry = _prompt_cache_compatibility_retry(
            retry,
            message,
            enabled=self.prompt_cache_mode != "provider_default",
        )
        if cache_retry is not None:
            return cache_retry
        return _generic_compatibility_retry(retry, message)

    def _completion_from_response(
        self,
        provider_call: _ProviderCallResult,
        *,
        additional_removed: tuple[str, ...] = (),
    ) -> LLMCompletion:
        response = provider_call.response
        error = getattr(response, "error", None)
        if error is not None:
            raise llm_provider_failure_error(
                error,
                diagnostic_type="ProviderResponseError",
            )
        status = _get_attr_or_key(response, "status")
        if status not in {None, "completed"}:
            details = _get_attr_or_key(response, "incomplete_details")
            reason = _get_attr_or_key(details, "reason") if details is not None else None
            raise llm_provider_failure_error(
                (status, reason),
                diagnostic_type="ProviderResponseStatus",
            )
        output = _bounded_provider_items(
            getattr(response, "output", None),
            limit=LLM_RESPONSE_OUTPUT_MAX_ITEMS,
            label="response.output",
        )
        tool_calls: list[dict[str, Any]] = []
        argument_chars = 0
        argument_bytes = 0
        for item in output:
            if _get_attr_or_key(item, "type") != "function_call":
                continue
            if len(tool_calls) >= LLM_RESPONSE_TOOL_CALL_MAX_COUNT:
                raise _provider_response_bounds_error("response.tool_calls")
            item_status = _get_attr_or_key(item, "status")
            if item_status not in {None, "completed"}:
                raise llm_provider_failure_error(
                    item_status,
                    diagnostic_type="ProviderFunctionCallStatus",
                )
            arguments = _get_attr_or_key(item, "arguments") or "{}"
            arguments = _bounded_provider_text(
                arguments,
                limit=LLM_RESPONSE_TOOL_ARGUMENT_MAX_CHARS,
                label="response.tool_call.arguments",
            )
            argument_chars += len(arguments)
            argument_bytes += len(arguments.encode("utf-8"))
            if argument_chars > LLM_RESPONSE_TOOL_ARGUMENT_TOTAL_MAX_CHARS:
                raise _provider_response_bounds_error(
                    "response.tool_call.arguments_total"
                )
            if argument_bytes > LLM_RESPONSE_TOOL_ARGUMENT_TOTAL_MAX_BYTES:
                raise _provider_response_bounds_error(
                    "response.tool_call.arguments_total_bytes"
                )
            tool_calls.append(
                {
                    "id": _get_attr_or_key(item, "id") or _get_attr_or_key(item, "call_id"),
                    "call_id": _get_attr_or_key(item, "call_id"),
                    "name": _get_attr_or_key(item, "name"),
                    "arguments": arguments,
                }
            )
        completion = LLMCompletion(
            content=self._response_text(response, output=output),
            tool_calls=tool_calls,
            raw=response,
            api="responses",
            response_id=getattr(response, "id", None),
            request_id=getattr(response, "_request_id", None),
            model=str(getattr(response, "model", "")) or None,
            usage=_usage_from_response(response),
            reasoning=_reasoning_from_response(response, output=output),
            provider_request_options=_provider_request_option_observation(
                provider_call.request
            ),
            compatibility_removed_options=sorted(
                set(additional_removed).union(
                    provider_call.compatibility_removed_options
                )
            ),
            _provider_attempt_sequence=provider_call.attempt_sequence,
        )
        _enrich_active_provider_trace(completion)
        return completion

    def _completion_from_chat(
        self,
        provider_call: _ProviderCallResult,
        *,
        additional_removed: tuple[str, ...] = (),
    ) -> LLMCompletion:
        completion = provider_call.response
        try:
            message = completion.choices[0].message
        except (AttributeError, IndexError) as exc:
            raise llm_provider_failure_error(exc) from exc

        raw_tool_calls = _bounded_provider_items(
            getattr(message, "tool_calls", None),
            limit=LLM_RESPONSE_TOOL_CALL_MAX_COUNT,
            label="chat.message.tool_calls",
        )
        tool_calls: list[dict[str, Any]] = []
        argument_chars = 0
        argument_bytes = 0
        for call in raw_tool_calls:
            function = getattr(call, "function", None)
            if function is None:
                continue
            arguments = getattr(function, "arguments", "{}") or "{}"
            arguments = _bounded_provider_text(
                arguments,
                limit=LLM_RESPONSE_TOOL_ARGUMENT_MAX_CHARS,
                label="chat.tool_call.arguments",
            )
            argument_chars += len(arguments)
            argument_bytes += len(arguments.encode("utf-8"))
            if argument_chars > LLM_RESPONSE_TOOL_ARGUMENT_TOTAL_MAX_CHARS:
                raise _provider_response_bounds_error(
                    "chat.tool_call.arguments_total"
                )
            if argument_bytes > LLM_RESPONSE_TOOL_ARGUMENT_TOTAL_MAX_BYTES:
                raise _provider_response_bounds_error(
                    "chat.tool_call.arguments_total_bytes"
                )
            tool_calls.append(
                {
                    "id": getattr(call, "id", None),
                    "name": getattr(function, "name", None),
                    "arguments": arguments,
                }
            )
        content = self._message_content(message)
        finish_reason = _first_choice_attr(completion, "finish_reason")
        if finish_reason in {"length", "content_filter"} and (tool_calls or content.strip()):
            raise llm_provider_failure_error(
                finish_reason,
                diagnostic_type="ProviderFinishReason",
            )
        if tool_calls and finish_reason not in {None, "tool_calls", "function_call"}:
            raise llm_provider_failure_error(
                finish_reason,
                diagnostic_type="ProviderFinishReason",
            )

        result = LLMCompletion(
            content=content,
            tool_calls=tool_calls,
            raw=completion,
            api="chat",
            response_id=getattr(completion, "id", None),
            request_id=getattr(completion, "_request_id", None),
            model=str(getattr(completion, "model", "")) or None,
            usage=_usage_from_response(completion),
            reasoning=_reasoning_from_chat_message(message),
            provider_request_options=_provider_request_option_observation(
                provider_call.request
            ),
            compatibility_removed_options=sorted(
                set(additional_removed).union(
                    provider_call.compatibility_removed_options
                )
            ),
            _provider_attempt_sequence=provider_call.attempt_sequence,
        )
        _enrich_active_provider_trace(result)
        return result

    def _use_responses_api(self) -> bool:
        if self.api_mode == "responses":
            return True
        if self.api_mode == "chat":
            return False
        return self.base_url is None or _is_openai_base_url(self.base_url)

    def _validate_base_url_policy(self) -> None:
        if self.base_url and not _is_openai_base_url(self.base_url) and not self.allow_custom_base_url:
            raise LLMError(
                "OPENAI_BASE_URL points to a custom endpoint; pass allow_custom_base_url=True "
                "or set AGENT_LIBOS_ALLOW_CUSTOM_LLM_BASE_URL=1 in the host environment"
            )

    def _text_config(self, json_mode: bool) -> dict[str, Any]:
        config: dict[str, Any] = {}
        if json_mode:
            config["format"] = {"type": "json_object"}
        if self.verbosity:
            config["verbosity"] = self.verbosity
        return config

    def _responses_text_config_for_schema(self, schema: dict[str, Any], schema_name: str) -> dict[str, Any]:
        config = self._text_config(json_mode=False)
        config["format"] = {
            "type": "json_schema",
            "name": self._schema_name(schema_name),
            "schema": self._strict_structured_output_schema(schema),
            "strict": True,
        }
        return config

    def _chat_response_format_for_schema(self, schema: dict[str, Any], schema_name: str) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self._schema_name(schema_name),
                "schema": self._strict_structured_output_schema(schema),
                "strict": True,
            },
        }

    def _strict_structured_output_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        try:
            return normalize_openai_structured_output_schema(schema)
        except ValueError as exc:
            raise LLMError(str(exc)) from exc

    @staticmethod
    def _schema_name(value: str) -> str:
        selected = str(value or "").strip()
        if not selected:
            raise LLMError("schema_name must be a non-empty string")
        return selected

    def _add_provider_request_options(self, payload: dict[str, Any]) -> None:
        """Add explicitly configured OpenAI-compatible request options.

        A custom base URL is an explicit host-selected provider boundary. When
        its profile opts into these fields, dispatch them and let the bounded
        compatibility retry remove only fields the provider rejects. Defaults
        remain unset, so no option is inferred from an endpoint hostname.
        """
        if self.safety_identifier:
            if len(self.safety_identifier) > 64:
                raise LLMError("safety_identifier must be at most 64 characters")
            payload["safety_identifier"] = self.safety_identifier
        if self.prompt_cache_key:
            payload["prompt_cache_key"] = self.prompt_cache_key
        if self.prompt_cache_mode != "provider_default":
            options: dict[str, str] = {"mode": str(self.prompt_cache_mode)}
            if self.prompt_cache_ttl is not None:
                options["ttl"] = self.prompt_cache_ttl
            payload["prompt_cache_options"] = options
        elif self.prompt_cache_retention:
            # Normalize again at the provider boundary so even a caller that
            # mutates the dataclass after construction can never dispatch the
            # retired hyphenated spelling.
            selected_retention = _normalize_prompt_cache_retention(
                self.prompt_cache_retention,
                label="prompt_cache_retention",
            )
            self.prompt_cache_retention = selected_retention
            payload["prompt_cache_retention"] = selected_retention

    def _finalize_prompt_cache_request(self, payload: dict[str, Any]) -> None:
        """Bind an opt-in cache key to the stable prefix and tool set.

        The configured key is a Host privacy/routing domain.  Hashing it with
        the exact marked prefix and normalized tools avoids per-Run identifiers
        while keeping unrelated prompt families off the same hot routing key.
        Legacy/provider-default behavior preserves the configured key verbatim.
        """

        host_stable = payload.pop(_CACHE_STABLE_PROJECTION_KEY, None)
        if self.prompt_cache_mode == "provider_default":
            return
        base_key = payload.get("prompt_cache_key")
        if not isinstance(base_key, str) or not base_key:
            raise LLMError(
                "prompt_cache_key is required for implicit or explicit prompt caching"
            )
        if base_key.startswith("alibos:v2:"):
            return
        stable = (
            host_stable
            if host_stable is not None
            else _stable_prompt_cache_projection(payload)
        )
        if stable is None:
            raise LLMError(
                "prompt cache mode requires one stable prompt_cache_breakpoint"
            )
        digest = hashlib.sha256(
            json.dumps(
                {
                    "version": 2,
                    "domain": base_key,
                    "provider": str(self.base_url or "https://api.openai.com/v1"),
                    "model": payload.get("model"),
                    "stable_prefix": stable,
                    "tools": payload.get("tools", []),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        payload["prompt_cache_key"] = f"alibos:v2:{digest[:54]}"

    def _use_openai_request_options(self) -> bool:
        return self.base_url is None or _is_openai_base_url(self.base_url)

    def _extra_body(self) -> dict[str, Any]:
        configured, enabled = self._enable_thinking_setting()
        if not configured:
            return {}
        return {"enable_thinking": enabled}

    def _needs_non_thinking_retry(self, completion: LLMCompletion) -> bool:
        if completion.tool_calls or completion.content.strip():
            return False
        configured, _enabled = self._enable_thinking_setting()
        if configured:
            return False
        if self.base_url is None or _is_openai_base_url(self.base_url):
            return False
        return True

    def _enable_thinking_setting(self) -> tuple[bool, bool]:
        if self.enable_thinking is not None:
            return True, bool(self.enable_thinking)
        if not self.inherit_ambient_openai_sdk_config:
            return False, False
        configured = os.getenv("OPENAI_ENABLE_THINKING")
        if configured is None:
            return False, False
        return True, _bool_env_value(configured)

    def _temperature(self, value: float | None) -> float:
        return self.defaults.temperature if value is None else value

    def _max_tokens(self, value: int | None) -> int:
        return self.defaults.max_tokens if value is None else value

    def _parallel_tool_calls(self, value: bool | None) -> bool:
        selected = self.parallel_tool_calls if value is None else value
        return bool(selected)

    def _should_fallback_to_chat(self, exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        message = str(exc).lower()
        if status_code in self.defaults.fallback_status_codes:
            return True
        return any(
            fragment in message
            for fragment in (
                "responses",
                "unknown url",
                "unsupported endpoint",
                "not found",
                "invalid endpoint",
            )
        )

    @staticmethod
    def _is_tool_protocol_rejection(exc: Exception) -> bool:
        message = str(exc).lower()
        return "tools" in message or "tool_choice" in message

    @staticmethod
    def _with_enable_thinking(payload: dict[str, Any], enabled: bool) -> dict[str, Any]:
        retry = dict(payload)
        extra_body = dict(retry.get("extra_body") or {})
        extra_body["enable_thinking"] = enabled
        retry["extra_body"] = extra_body
        return retry

    @staticmethod
    def _response_text(
        response: Any,
        *,
        output: list[Any] | None = None,
    ) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return _bounded_provider_text(
                output_text,
                limit=LLM_RESPONSE_CONTENT_MAX_CHARS,
                label="response.output_text",
            )
        parts: list[str] = []
        total_chars = 0
        total_bytes = 0
        selected_output = output
        if selected_output is None:
            selected_output = _bounded_provider_items(
                getattr(response, "output", None),
                limit=LLM_RESPONSE_OUTPUT_MAX_ITEMS,
                label="response.output",
            )
        for item in selected_output:
            if _get_attr_or_key(item, "type") != "message":
                continue
            contents = _bounded_provider_items(
                _get_attr_or_key(item, "content"),
                limit=LLM_RESPONSE_CONTENT_MAX_PARTS,
                label="response.message.content",
            )
            for content in contents:
                if _get_attr_or_key(content, "type") == "output_text":
                    text = _bounded_provider_text(
                        _get_attr_or_key(content, "text") or "",
                        limit=LLM_RESPONSE_CONTENT_MAX_CHARS,
                        label="response.message.content.text",
                    )
                    total_chars += len(text)
                    total_bytes += len(text.encode("utf-8"))
                    if total_chars > LLM_RESPONSE_CONTENT_MAX_CHARS:
                        raise _provider_response_bounds_error(
                            "response.content_total"
                        )
                    if total_bytes > LLM_RESPONSE_TEXT_MAX_BYTES:
                        raise _provider_response_bounds_error(
                            "response.content_total_bytes"
                        )
                    parts.append(text)
        return "".join(parts)

    @staticmethod
    def _message_content(message: Any) -> str:
        content = getattr(message, "content", None)
        if content is None:
            return ""
        if isinstance(content, str):
            return _bounded_provider_text(
                content,
                limit=LLM_RESPONSE_CONTENT_MAX_CHARS,
                label="chat.message.content",
            )
        if isinstance(content, list):
            parts = _bounded_provider_items(
                content,
                limit=LLM_RESPONSE_CONTENT_MAX_PARTS,
                label="chat.message.content",
            )
            selected: list[str] = []
            total_chars = 0
            total_bytes = 0
            for part in parts:
                text = _provider_content_part_text(part)
                total_chars += len(text)
                total_bytes += len(text.encode("utf-8"))
                if total_chars > LLM_RESPONSE_CONTENT_MAX_CHARS:
                    raise _provider_response_bounds_error(
                        "chat.message.content_total"
                    )
                if total_bytes > LLM_RESPONSE_TEXT_MAX_BYTES:
                    raise _provider_response_bounds_error(
                        "chat.message.content_total_bytes"
                    )
                selected.append(text)
            return "".join(selected)
        raise _provider_response_bounds_error("chat.message.content_type")

    def _messages_with_json_instruction(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        json_instruction = self.defaults.json_instruction
        if _static_messages_contain_instruction(messages, json_instruction):
            return messages
        new_messages = [dict(message) for message in messages]
        for msg in new_messages:
            if msg.get("role") in {"system", "developer"}:
                msg["content"] = str(msg.get("content", "")) + f" {json_instruction}"
                return new_messages
        return [{"role": "system", "content": json_instruction}] + new_messages


def _responses_tools_from_chat_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        converted_tool = openai_responses_tool_schema(tool)
        if converted_tool is not None:
            converted.append(converted_tool)
    return converted


def _chat_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_openai_chat_tool_schema(tool) for tool in tools]


def _tools_without_strict(tools: list[Any]) -> list[Any]:
    selected: list[Any] = []
    for tool in tools:
        if not isinstance(tool, dict):
            selected.append(tool)
            continue
        updated = dict(tool)
        updated.pop("strict", None)
        function = updated.get("function")
        if isinstance(function, dict):
            updated["function"] = dict(function)
            updated["function"].pop("strict", None)
        selected.append(updated)
    return selected


def _llm_defaults(config: AgentLibOSConfig | LLMDefaults | None) -> LLMDefaults:
    if config is None:
        return DEFAULT_CONFIG.llm
    if isinstance(config, LLMDefaults):
        return config
    return config.llm


@contextmanager
def _provider_attempt_kind(kind: ProviderAttemptKind) -> Any:
    token = _ACTIVE_PROVIDER_ATTEMPT_KIND.set(kind)
    try:
        yield
    finally:
        _ACTIVE_PROVIDER_ATTEMPT_KIND.reset(token)


def _enrich_active_provider_trace(completion: LLMCompletion) -> None:
    trace = _ACTIVE_PROVIDER_TRACE.get()
    sequence = completion._provider_attempt_sequence
    if trace is None or sequence is None:
        return
    try:
        trace.enrich_response(
            sequence,
            reasoning=completion.reasoning,
            output=completion.content,
            tool_calls=completion.tool_calls,
            usage=completion.usage,
            model=completion.model,
            request_id=completion.request_id,
            response_id=completion.response_id,
        )
    except Exception:
        # Trace construction is diagnostic and must never turn a valid Provider
        # completion into a second logical call or a failed tool selection.
        trace.limited = True


def _reject_active_provider_attempt(
    provider_call: _ProviderCallResult,
    error: BaseException,
) -> None:
    _reject_active_provider_sequence(provider_call.attempt_sequence, error)


def _reject_active_provider_sequence(
    sequence: int | None,
    error: BaseException,
) -> None:
    trace = _ACTIVE_PROVIDER_TRACE.get()
    if trace is None or sequence is None:
        return
    try:
        trace.reject_response(sequence, error)
    except Exception:
        trace.limited = True


def _prompt_cache_compatibility_retry(
    payload: dict[str, Any],
    message: str,
    *,
    enabled: bool,
) -> dict[str, Any] | None:
    cache_option_error = any(
        option in message
        for option in (
            "prompt_cache_key",
            "prompt_cache_retention",
            "prompt_cache_options",
            "prompt_cache_breakpoint",
        )
    )
    if not cache_option_error or not enabled:
        return None
    retry = dict(payload)
    retry.pop("prompt_cache_key", None)
    retry.pop("prompt_cache_retention", None)
    retry.pop("prompt_cache_options", None)
    return _without_prompt_cache_breakpoints(retry)


def _generic_compatibility_retry(
    payload: dict[str, Any],
    message: str,
) -> dict[str, Any] | None:
    retry = dict(payload)
    for key in (
        "parallel_tool_calls",
        "response_format",
        "temperature",
        "store",
        "reasoning",
        "reasoning_effort",
        "previous_response_id",
        "prompt_cache_key",
        "prompt_cache_retention",
        "prompt_cache_options",
        "safety_identifier",
    ):
        if key in message and key in retry:
            retry.pop(key, None)
            return retry
    if "verbosity" in message:
        if "verbosity" in retry:
            retry.pop("verbosity", None)
            return retry
        text = retry.get("text")
        if isinstance(text, dict) and "verbosity" in text:
            updated_text = dict(text)
            updated_text.pop("verbosity", None)
            retry["text"] = updated_text
            return retry
    if (
        any(option in message for option in ("text", "json_schema", "json_object"))
        and "text" in retry
    ):
        retry.pop("text", None)
        return retry
    if "response_format" in message and "response_format" in retry:
        retry.pop("response_format", None)
        return retry
    return None


def _messages_for_provider(
    messages: list[dict[str, Any]],
    *,
    api: Literal["responses", "chat"],
    add_cache_breakpoint: bool,
) -> list[dict[str, Any]]:
    """Remove Host-only message metadata and mark one stable text block."""

    selected, explicit_target, prefix_chars, leading_static_target = (
        _prepare_provider_messages(messages)
    )
    if not add_cache_breakpoint or not selected:
        return selected
    target = explicit_target if explicit_target is not None else leading_static_target
    if target is None:
        return selected
    _attach_cache_breakpoint(
        selected[target],
        api=api,
        stable_prefix_chars=prefix_chars.get(target),
    )
    return selected


def _prepare_provider_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int | None, dict[int, int], int | None]:
    selected: list[dict[str, Any]] = []
    explicit_target: int | None = None
    stable_prefix_chars: dict[int, int] = {}
    leading_static_target: int | None = None
    leading_static = True
    for index, message in enumerate(messages):
        copied = {
            key: value
            for key, value in message.items()
            if key not in {
                _CACHE_STABLE_MESSAGE_KEY,
                _CACHE_STABLE_PREFIX_CHARS_KEY,
            }
        }
        selected.append(copied)
        prefix_chars = message.get(_CACHE_STABLE_PREFIX_CHARS_KEY)
        if (
            isinstance(prefix_chars, int)
            and not isinstance(prefix_chars, bool)
            and prefix_chars > 0
        ):
            stable_prefix_chars[index] = prefix_chars
            explicit_target = index
        elif message.get(_CACHE_STABLE_MESSAGE_KEY) is True:
            explicit_target = index
        role = str(message.get("role") or "user")
        if leading_static and role in {"system", "developer"}:
            leading_static_target = index
        else:
            leading_static = False
    return selected, explicit_target, stable_prefix_chars, leading_static_target


def _attach_cache_breakpoint(
    message: dict[str, Any],
    *,
    api: Literal["responses", "chat"],
    stable_prefix_chars: int | None,
) -> None:
    content = message.get("content")
    block_type = "input_text" if api == "responses" else "text"
    breakpoint = {"mode": "explicit"}
    if isinstance(content, str):
        if not content:
            return
        if stable_prefix_chars is not None and stable_prefix_chars < len(content):
            message["content"] = [
                {
                    "type": block_type,
                    "text": content[:stable_prefix_chars],
                    "prompt_cache_breakpoint": breakpoint,
                },
                {
                    "type": block_type,
                    "text": content[stable_prefix_chars:],
                },
            ]
            return
        message["content"] = [
            {
                "type": block_type,
                "text": content,
                "prompt_cache_breakpoint": breakpoint,
            }
        ]
        return
    if not isinstance(content, list):
        return
    copied_parts = [dict(part) if isinstance(part, dict) else part for part in content]
    for part_index in range(len(copied_parts) - 1, -1, -1):
        part = copied_parts[part_index]
        if not isinstance(part, dict) or not isinstance(part.get("text"), str):
            continue
        part = dict(part)
        part["type"] = block_type
        part["prompt_cache_breakpoint"] = breakpoint
        copied_parts[part_index] = part
        message["content"] = copied_parts
        return


def _host_stable_message_projection(
    messages: list[dict[str, Any]],
) -> dict[str, Any] | list[dict[str, Any]] | None:
    """Return the routing-stable prompt family without private markers."""

    explicit_target: int | None = None
    leading_static_target: int | None = None
    leading_static = True
    selected: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        selected.append(
            {
                key: value
                for key, value in message.items()
                if key not in {
                    _CACHE_STABLE_MESSAGE_KEY,
                    _CACHE_STABLE_PREFIX_CHARS_KEY,
                }
            }
        )
        prefix_chars = message.get(_CACHE_STABLE_PREFIX_CHARS_KEY)
        if (
            isinstance(prefix_chars, int)
            and not isinstance(prefix_chars, bool)
            and prefix_chars > 0
        ):
            explicit_target = index
        elif message.get(_CACHE_STABLE_MESSAGE_KEY) is True:
            explicit_target = index
        role = str(message.get("role") or "user")
        if leading_static and role in {"system", "developer"}:
            leading_static_target = index
        else:
            leading_static = False
    if leading_static_target is not None:
        return {
            "leading_instructions": selected[: leading_static_target + 1],
        }
    if explicit_target is None:
        return None
    return selected[: explicit_target + 1]


def _messages_to_responses_parts(
    messages: list[dict[str, Any]],
    *,
    native_tool_outputs: bool = False,
    tool_output_max_chars: int = 0,
) -> tuple[str | None, list[dict[str, Any]]]:
    instructions: list[str] = []
    input_items: list[dict[str, Any]] = []
    represented_call_ids: set[str] = set()
    for message in messages:
        role = str(message.get("role", "user"))
        raw_content = message.get("content")
        content = _message_content_for_search(message)
        if role in {"system", "developer"}:
            if _content_has_prompt_cache_breakpoint(raw_content):
                input_items.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": raw_content,
                    }
                )
                continue
            if content:
                instructions.append(content)
            continue
        if role == "tool":
            call_id = message.get("tool_call_id") or message.get("call_id")
            if call_id and (
                native_tool_outputs or str(call_id) in represented_call_ids
            ):
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(call_id),
                        "output": _bounded_tool_output(
                            content,
                            max_chars=tool_output_max_chars,
                        ),
                    }
                )
                continue
            input_items.append(
                {
                    "role": "user",
                    "content": _plain_tool_output_context(
                        message,
                        content,
                        max_chars=tool_output_max_chars,
                    ),
                }
            )
            continue
        if role == "assistant":
            assistant_items, assistant_call_ids = _responses_assistant_items(
                message,
                content,
            )
            input_items.extend(assistant_items)
            represented_call_ids.update(assistant_call_ids)
            if content or message.get("tool_calls"):
                continue
        input_items.append(
            {
                "role": "assistant" if role == "assistant" else "user",
                "content": (
                    raw_content
                    if _content_has_prompt_cache_breakpoint(raw_content)
                    else content
                ),
            }
        )
    return ("\n\n".join(instructions) if instructions else None), input_items


def _content_has_prompt_cache_breakpoint(value: Any) -> bool:
    return isinstance(value, list) and any(
        isinstance(part, dict) and "prompt_cache_breakpoint" in part
        for part in value
    )


def _prompt_cache_breakpoint_count(payload: dict[str, Any]) -> int:
    count = 0
    for container_key in ("messages", "input"):
        entries = payload.get(container_key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            content = entry.get("content")
            if isinstance(content, list):
                count += sum(
                    1
                    for part in content
                    if isinstance(part, dict)
                    and "prompt_cache_breakpoint" in part
                )
    return count


def _without_prompt_cache_breakpoints(payload: dict[str, Any]) -> dict[str, Any]:
    selected = dict(payload)
    for container_key in ("messages", "input"):
        entries = selected.get(container_key)
        if not isinstance(entries, list):
            continue
        updated_entries: list[Any] = []
        for entry in entries:
            if not isinstance(entry, dict):
                updated_entries.append(entry)
                continue
            updated_entry = dict(entry)
            content = updated_entry.get("content")
            if isinstance(content, list):
                updated_content: list[Any] = []
                for part in content:
                    if isinstance(part, dict):
                        updated_part = dict(part)
                        updated_part.pop("prompt_cache_breakpoint", None)
                        updated_content.append(updated_part)
                    else:
                        updated_content.append(part)
                updated_entry["content"] = updated_content
            updated_entries.append(updated_entry)
        selected[container_key] = updated_entries
    return selected


def _stable_prompt_cache_projection(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return the invariant routing family, falling back to the breakpoint."""

    instructions = payload.get("instructions")
    for container_key in ("messages", "input"):
        entries = payload.get(container_key)
        if not isinstance(entries, list):
            continue
        leading: list[Any] = []
        for entry in entries:
            if not isinstance(entry, dict) or str(entry.get("role") or "") not in {
                "system",
                "developer",
            }:
                break
            leading.append(entry)
        if instructions not in (None, "", [], {}) or leading:
            return {
                "instructions": instructions,
                "leading_messages": leading,
            }

    for container_key in ("messages", "input"):
        entries = payload.get(container_key)
        if not isinstance(entries, list):
            continue
        prefix: list[Any] = []
        for entry in entries:
            if not isinstance(entry, dict):
                prefix.append(entry)
                continue
            content = entry.get("content")
            if not isinstance(content, list):
                prefix.append(entry)
                continue
            before: list[Any] = []
            found = False
            for part in content:
                before.append(part)
                if isinstance(part, dict) and "prompt_cache_breakpoint" in part:
                    found = True
                    break
            if found:
                marked_entry = dict(entry)
                marked_entry["content"] = before
                prefix.append(marked_entry)
                return {
                    "instructions": payload.get("instructions"),
                    "container": container_key,
                    "prefix": prefix,
                }
            prefix.append(entry)
    return None


def _responses_assistant_items(
    message: dict[str, Any],
    content: str,
) -> tuple[list[dict[str, Any]], set[str]]:
    items: list[dict[str, Any]] = []
    call_ids: set[str] = set()
    if content:
        items.append({"role": "assistant", "content": content})
    for tool_call in list(message.get("tool_calls") or []):
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        name = (
            function.get("name")
            if isinstance(function, dict)
            else tool_call.get("name")
        )
        arguments = (
            function.get("arguments", "{}")
            if isinstance(function, dict)
            else tool_call.get("arguments", "{}")
        )
        call_id = tool_call.get("id") or tool_call.get("call_id")
        if not call_id or not name:
            continue
        selected_call_id = str(call_id)
        call_ids.add(selected_call_id)
        items.append(
            {
                "type": "function_call",
                "call_id": selected_call_id,
                "name": str(name),
                "arguments": (
                    arguments
                    if isinstance(arguments, str)
                    else json.dumps(arguments, sort_keys=True)
                ),
            }
        )
    return items, call_ids


def _plain_tool_output_context(message: dict[str, Any], content: str, *, max_chars: int) -> str:
    call_id = message.get("tool_call_id") or message.get("call_id")
    name = message.get("name") or message.get("tool_name")
    labels: list[str] = []
    if call_id:
        labels.append(f"call_id={call_id}")
    if name:
        labels.append(f"name={name}")
    header = "Tool output"
    if labels:
        header += f" ({', '.join(str(label) for label in labels)})"
    prefix = f"{header}:\n"
    if len(prefix) >= max_chars:
        prefix = "Tool output:\n"
    if len(prefix) >= max_chars:
        return prefix[:max_chars]
    return prefix + _bounded_tool_output(
        content,
        max_chars=max(0, max_chars - len(prefix)),
    )


def _bounded_tool_output(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 0:
        return ""

    # Recompute once because the omitted count can change the marker width.
    included_chars = max_chars
    marker = ""
    for _ in range(2):
        omitted_chars = len(value) - included_chars
        marker = (
            "[tool_output_omitted: "
            f"original_chars={len(value)} "
            f"included_chars={included_chars} "
            f"omitted_chars={omitted_chars}]"
        )
        included_chars = max(0, max_chars - len(marker) - 1)
    omitted_chars = len(value) - included_chars
    marker = (
        "[tool_output_omitted: "
        f"original_chars={len(value)} "
        f"included_chars={included_chars} "
        f"omitted_chars={omitted_chars}]"
    )
    if len(marker) >= max_chars:
        # Configuration validation keeps the normal limit comfortably above
        # the marker. This branch still enforces the hard model-facing bound
        # for direct construction with an unusually small custom default.
        return marker[:max_chars]
    return f"{value[:included_chars]}\n{marker}"


def _messages_have_unrepresentable_tool_output(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if str(message.get("role", "user")) != "tool":
            continue
        if message.get("tool_call_id") or message.get("call_id"):
            continue
        return True
    return False


def _provider_request_option_observation(
    request: dict[str, Any],
) -> dict[str, Any]:
    """Return non-secret facts about the request that actually succeeded."""

    cache_options = request.get("prompt_cache_options")
    cache_mode = (
        cache_options.get("mode")
        if isinstance(cache_options, dict)
        else "provider_default"
    )
    cache_ttl = (
        cache_options.get("ttl")
        if isinstance(cache_options, dict)
        else None
    )
    observation = {
        "prompt_cache_key_sent": "prompt_cache_key" in request,
        "prompt_cache_retention": request.get("prompt_cache_retention"),
        "prompt_cache_options_sent": "prompt_cache_options" in request,
        "safety_identifier_sent": "safety_identifier" in request,
    }
    breakpoint_count = _prompt_cache_breakpoint_count(request)
    if "prompt_cache_options" in request or breakpoint_count:
        observation.update(
            {
                "prompt_cache_mode": cache_mode,
                "prompt_cache_ttl": cache_ttl,
                "prompt_cache_breakpoint_count": breakpoint_count,
            }
        )
    return observation


def _is_openai_sdk_error(exc: Exception) -> bool:
    try:
        from openai import OpenAIError
    except ImportError:
        return False
    return isinstance(exc, OpenAIError)


def llm_provider_failure_error(
    source: BaseException | object,
    *,
    transient: bool = False,
    diagnostic_type: str | None = None,
) -> LLMError:
    """Wrap provider-authored failure data in one text-free public error.

    The raw exception remains available as ``__cause__`` at the raise site.
    Durable callers may recover only the private type/length/digest observation
    through :func:`llm_error_internal_observation`; no provider text is copied
    into the wrapper message or attributes.
    """

    if isinstance(source, LLMError):
        try:
            marker = object.__getattribute__(
                source,
                _PROVIDER_FAILURE_MARKER_ATTR,
            )
        except BaseException:
            marker = None
        if marker is _PROVIDER_FAILURE_MARKER:
            return source

    error_class = (
        LLMTransientError
        if transient or isinstance(source, LLMTransientError)
        else LLMError
    )
    wrapped = error_class("LLM provider failure")
    public_error = public_error_envelope(wrapped, code="llm_error")
    # Cache the public envelope before replacing ``args`` so every downstream
    # boundary observes exactly the same correlation identifier.
    wrapped.args = (public_error["message"],)
    if isinstance(source, BaseException):
        observation = internal_exception_observation(
            source,
            correlation_id=public_error["correlation_id"],
        )
    else:
        observation = internal_error_observation(
            error_type=diagnostic_type or type(source).__name__,
            text=_safe_provider_diagnostic_text(source),
            correlation_id=public_error["correlation_id"],
        )
    object.__setattr__(
        wrapped,
        _PROVIDER_FAILURE_MARKER_ATTR,
        _PROVIDER_FAILURE_MARKER,
    )
    object.__setattr__(
        wrapped,
        _PROVIDER_FAILURE_OBSERVATION_ATTR,
        observation,
    )
    return wrapped


def llm_error_internal_observation(
    error: BaseException,
    *,
    correlation_id: str,
) -> dict[str, Any]:
    """Return a provider-source observation when ``error`` is a safe wrapper."""

    try:
        marker = object.__getattribute__(error, _PROVIDER_FAILURE_MARKER_ATTR)
        selected = object.__getattribute__(
            error,
            _PROVIDER_FAILURE_OBSERVATION_ATTR,
        )
    except BaseException:
        marker = None
        selected = None
    if marker is _PROVIDER_FAILURE_MARKER and isinstance(selected, dict):
        text = selected.get("exception_text")
        if (
            selected.get("correlation_id") == correlation_id
            and isinstance(selected.get("error_type"), str)
            and isinstance(text, dict)
            and type(text.get("bytes")) is int
            and text["bytes"] >= 0
            and isinstance(text.get("sha256"), str)
            and len(text["sha256"]) == 64
        ):
            return {
                "error_type": selected["error_type"],
                "correlation_id": correlation_id,
                "exception_text": {
                    "bytes": text["bytes"],
                    "sha256": text["sha256"],
                },
            }
    return internal_exception_observation(
        error,
        correlation_id=correlation_id,
    )


def _safe_provider_diagnostic_text(value: object) -> str:
    try:
        return str(value)
    except BaseException:
        return "provider diagnostic text is unavailable"


def _openai_sdk_request_error(exc: Exception) -> LLMError:
    return llm_provider_failure_error(
        exc,
        transient=_is_transient_openai_sdk_error(exc),
    )


def _is_transient_openai_sdk_error(exc: Exception) -> bool:
    """Mirror the SDK's documented retryable transport/status classes."""

    return _should_retry_openai_sdk_error(exc)


def _should_retry_openai_sdk_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and 300 <= status_code <= 399:
        # Redirects are disabled at the transport and remain terminal even if a
        # compatible endpoint tries to override retry policy with a header.
        return False
    retry_directive = _openai_error_header(exc, "x-should-retry")
    if retry_directive is not None:
        normalized = retry_directive.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False

    if isinstance(status_code, int) and (
        status_code in {408, 409, 429} or 500 <= status_code <= 599
    ):
        return True
    return type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }


def _openai_retry_delay(exc: Exception, retry_index: int) -> float:
    retry_after_ms = _openai_error_header(exc, "retry-after-ms")
    if retry_after_ms is not None:
        try:
            selected_ms = float(retry_after_ms)
        except (TypeError, ValueError):
            selected_ms = -1.0
        selected_seconds = selected_ms / 1000.0
        if 0.0 <= selected_seconds <= 60.0:
            return selected_seconds

    retry_after = _openai_error_header(exc, "retry-after")
    if retry_after is not None:
        try:
            selected_seconds = float(retry_after)
        except (TypeError, ValueError):
            selected_seconds = _retry_after_date_seconds(retry_after)
        if 0.0 <= selected_seconds <= 60.0:
            return selected_seconds

    exponential = min(8.0, 0.5 * (2 ** max(0, retry_index)))
    jittered = exponential * (0.75 + (0.5 * random.random()))
    return min(8.0, max(0.5, jittered))


def _retry_after_date_seconds(value: str) -> float:
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return -1.0
    if parsed is None:
        return -1.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (parsed - datetime.now(timezone.utc)).total_seconds()


def _openai_error_header(exc: Exception, name: str) -> str | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get(name)
    except (AttributeError, TypeError):
        return None
    return str(value) if value is not None else None



def _is_openai_base_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    if parsed.scheme and parsed.scheme != "https":
        return False
    host = parsed.hostname if parsed.scheme else urlparse(f"https://{base_url}").hostname
    return host == "api.openai.com"


def _static_messages_contain_instruction(
    messages: list[dict[str, Any]],
    instruction: str,
) -> bool:
    selected = instruction.strip()
    if not selected:
        return True
    return any(
        str(message.get("role", "user")) in {"system", "developer"}
        and selected in _message_content_for_search(message)
        for message in messages
    )


def _message_content_for_search(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(_content_part_text(part) for part in content)
    return "" if content is None else str(content)


def _provider_response_bounds_error(label: str) -> LLMError:
    return llm_provider_failure_error(
        label,
        diagnostic_type="ProviderResponseBounds",
    )


def _bounded_provider_text(value: Any, *, limit: int, label: str) -> str:
    if not isinstance(value, str) or len(value) > limit:
        raise _provider_response_bounds_error(label)
    try:
        encoded_bytes = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise _provider_response_bounds_error(f"{label}_encoding") from exc
    if encoded_bytes > LLM_RESPONSE_TEXT_MAX_BYTES:
        raise _provider_response_bounds_error(f"{label}_bytes")
    return value


def _bounded_provider_items(value: Any, *, limit: int, label: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray, dict)):
        raise _provider_response_bounds_error(f"{label}_type")
    if isinstance(value, (list, tuple)):
        if len(value) > limit:
            raise _provider_response_bounds_error(label)
        return list(value)
    try:
        iterator = iter(value)
    except TypeError as exc:
        raise _provider_response_bounds_error(f"{label}_type") from exc
    selected: list[Any] = []
    for item in iterator:
        if len(selected) >= limit:
            raise _provider_response_bounds_error(label)
        selected.append(item)
    return selected


def _provider_content_part_text(part: Any) -> str:
    if isinstance(part, dict):
        value = part.get("text") or part.get("content") or ""
    else:
        value = getattr(part, "text", getattr(part, "content", "")) or ""
    return _bounded_provider_text(
        value,
        limit=LLM_RESPONSE_CONTENT_MAX_CHARS,
        label="chat.message.content_part",
    )


def _content_part_text(part: Any) -> str:
    if isinstance(part, dict):
        return str(part.get("text") or part.get("content") or "")
    return str(getattr(part, "text", getattr(part, "content", part)) or "")


def _get_attr_or_key(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _first_choice_attr(completion: Any, attr: str) -> Any:
    try:
        return getattr(completion.choices[0], attr, None)
    except (AttributeError, IndexError):
        return None


def _usage_from_response(response: Any) -> dict[str, Any]:
    usage = _get_attr_or_key(response, "usage")
    if usage is None:
        return {}
    jsonable = to_jsonable(usage)
    return jsonable if isinstance(jsonable, dict) else {"raw": jsonable}


def _reasoning_from_response(
    response: Any,
    *,
    output: list[Any] | None = None,
) -> Any | None:
    # ``Response.reasoning`` is the request/configuration object in current
    # OpenAI SDKs. The actual Provider-authored reasoning is emitted as ordered
    # output items and must not be shadowed by that top-level setting.
    reasoning_items: list[Any] = []
    selected_output = output
    if selected_output is None:
        selected_output = _bounded_provider_items(
            _get_attr_or_key(response, "output"),
            limit=LLM_RESPONSE_OUTPUT_MAX_ITEMS,
            label="response.output",
        )
    for item in selected_output:
        if _get_attr_or_key(item, "type") == "reasoning":
            projected = project_provider_raw_response(item)
            if isinstance(projected, dict):
                if projected.get("type") == "omitted":
                    reasoning_items.append(projected)
                    continue
                selected_keys = {
                    "type",
                    "summary",
                    "content",
                    "encrypted_content",
                    "signature",
                }
                selected = {
                    key: value
                    for key, value in projected.items()
                    if key in selected_keys
                    or (
                        isinstance(value, dict)
                        and value.get("type") == "opaque"
                    )
                }
                reasoning_items.append(selected)
                if projected.get("_provider_projection_limited") is True:
                    omitted = projected.get("_omitted")
                    if isinstance(omitted, dict) and omitted.get("type") == "omitted":
                        reasoning_items.append(omitted)
                    else:
                        reasoning_items.append(
                            {
                                "type": "omitted",
                                "reason": "structure_limit",
                                "bytes": 0,
                                "sha256": hashlib.sha256(b"").hexdigest(),
                                "digest_scope": "bounded_summary",
                            }
                        )
    return reasoning_items or None


def _reasoning_from_chat_message(message: Any) -> Any | None:
    allowed = ("reasoning", "reasoning_content", "thinking", "thinking_content")
    selected: dict[str, Any] = {}
    for key in allowed:
        value = _get_attr_or_key(message, key)
        if _has_value(value):
            selected[key] = project_provider_raw_response(value)
    additional = _get_attr_or_key(message, "additional_kwargs")
    if isinstance(additional, dict):
        for key in allowed:
            if key in selected:
                continue
            value = additional.get(key)
            if _has_value(value):
                selected[key] = project_provider_raw_response(value)
    if not selected:
        return None
    return {key: selected[key] for key in allowed if key in selected}


def _has_value(value: Any) -> bool:
    return value is not None and value != ""


def _bool_env_value(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise LLMError(f"invalid boolean environment value: {value!r}")


def _bool_env(name: str, default: bool) -> bool:
    return _bool_env_from(os.environ, name, default=default)


def _bool_env_from(env: dict[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value is None or not value.strip():
        return default
    return _bool_env_value(value)


def _float_env(name: str, default: float) -> float:
    return _float_env_from(os.environ, name, default=default)


def _float_env_from(env: dict[str, str], name: str, default: float) -> float:
    value = env.get(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise LLMError(f"{name} must be a float, got {value!r}") from exc


def _int_env(name: str, default: int) -> int:
    return _int_env_from(os.environ, name, default=default)


def _int_env_from(env: dict[str, str], name: str, default: int) -> int:
    value = env.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise LLMError(f"{name} must be an integer, got {value!r}") from exc


def _optional_env(name: str) -> str | None:
    return _optional_env_from(os.environ, name)


def _optional_env_from(env: dict[str, str], name: str) -> str | None:
    value = env.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _verbosity_env(name: str) -> Literal["low", "medium", "high"] | None:
    return _verbosity_env_from(os.environ, name)


def _verbosity_env_from(env: dict[str, str], name: str) -> Literal["low", "medium", "high"] | None:
    value = _optional_env_from(env, name)
    if value is None:
        return None
    normalized = value.lower()
    if normalized not in {"low", "medium", "high"}:
        raise LLMError(f"{name} must be one of low, medium, high; got {value!r}")
    return normalized  # type: ignore[return-value]


def _prompt_cache_retention_env_from(env: dict[str, str], name: str) -> Literal["in_memory", "24h"] | None:
    value = _optional_env_from(env, name)
    if value is None:
        return None
    return _normalize_prompt_cache_retention(value, label=name)


def _prompt_cache_mode_env_from(
    env: dict[str, str],
    name: str,
) -> Literal["provider_default", "implicit", "explicit"] | None:
    value = _optional_env_from(env, name)
    if value is None:
        return None
    return _normalize_prompt_cache_mode(value, label=name)


def _prompt_cache_ttl_env_from(
    env: dict[str, str],
    name: str,
) -> Literal["30m"] | None:
    value = _optional_env_from(env, name)
    if value is None:
        return None
    return _normalize_prompt_cache_ttl(value, label=name)


def _normalize_prompt_cache_retention(
    value: str | None,
    *,
    label: str,
) -> Literal["in_memory", "24h"] | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized == "in-memory":
        normalized = "in_memory"
    if normalized not in {"in_memory", "24h"}:
        raise LLMError(f"{label} must be one of in_memory, 24h; got {value!r}")
    return normalized  # type: ignore[return-value]


def _normalize_prompt_cache_mode(
    value: str,
    *,
    label: str,
) -> Literal["provider_default", "implicit", "explicit"]:
    normalized = str(value).strip().lower()
    if normalized not in _PROMPT_CACHE_MODES:
        raise LLMError(
            f"{label} must be one of provider_default, implicit, explicit; got {value!r}"
        )
    return normalized  # type: ignore[return-value]


def _normalize_prompt_cache_ttl(
    value: str | None,
    *,
    label: str,
) -> Literal["30m"] | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized != "30m":
        raise LLMError(f"{label} must be 30m or unset; got {value!r}")
    return "30m"


def _validate_prompt_cache_options(
    *,
    mode: str,
    key: str | None,
    retention: str | None,
    ttl: str | None,
) -> None:
    if retention is not None and ttl is not None:
        raise LLMError(
            "prompt_cache_retention and prompt_cache_ttl are mutually exclusive"
        )
    if mode != "provider_default" and not str(key or "").strip():
        raise LLMError(
            "prompt_cache_key is required for implicit or explicit prompt caching"
        )
    if mode == "provider_default" and ttl is not None:
        raise LLMError(
            "prompt_cache_ttl requires implicit or explicit prompt_cache_mode"
        )
    if mode != "provider_default" and retention is not None:
        raise LLMError(
            "prompt_cache_retention cannot be combined with implicit or explicit mode"
        )


def _run_sync(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    if inspect.iscoroutine(awaitable):
        awaitable.close()
    raise RuntimeError("Cannot use sync LLMClient APIs inside a running event loop. Use async APIs instead.")


def _run_close_sync(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    result: dict[str, Any] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(awaitable)
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=runner, name="agent-libos-llm-close", daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


def read_dotenv(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def load_dotenv(path: str | Path = ".env") -> None:
    for key, value in read_dotenv(path).items():
        os.environ.setdefault(key, value)
