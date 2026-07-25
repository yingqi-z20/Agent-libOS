from __future__ import annotations

import asyncio
import inspect
import json
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig, LLMDefaults
from agent_libos.llm.openai_schema import (
    normalize_openai_chat_tool_schema,
    normalize_openai_structured_output_schema,
    openai_responses_tool_schema,
)
from agent_libos.models.exceptions import LibOSError
from agent_libos.ports.blocking_work import run_blocking_once
from agent_libos.utils.serde import to_jsonable

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_API_MODES = {"auto", "responses", "chat"}


class LLMError(LibOSError):
    pass


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


@dataclass(frozen=True)
class _ProviderCallResult:
    response: Any
    request: dict[str, Any]
    compatibility_removed_options: tuple[str, ...] = ()


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
        selected_messages = self._messages_with_json_instruction(messages) if json_mode and json_schema is None else messages
        completion = await self._complete_without_tools(
            messages=selected_messages,
            temperature=self._temperature(temperature),
            max_tokens=self._max_tokens(max_tokens),
            json_mode=json_mode,
            json_schema=json_schema,
            schema_name=schema_name,
        )
        if not completion.content:
            raise LLMError("LLM returned empty content")
        return completion

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
                elif self.fallback_json_actions and self._is_tool_protocol_rejection(exc):
                    return await self._complete_json_action_fallback(
                        messages,
                        selected_temperature,
                        selected_max_tokens,
                    )
                else:
                    raise
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
        return self._completion_from_response(provider_call)

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
        return self._completion_from_response(provider_call)

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
        result = self._completion_from_chat(provider_call)
        if self._needs_non_thinking_retry(result):
            previous_removed = provider_call.compatibility_removed_options
            retry_payload = self._with_enable_thinking(provider_call.request, enabled=False)
            retry_call = await self._create_chat_completion(retry_payload)
            result = self._completion_from_chat(
                retry_call,
                additional_removed=previous_removed,
            )
            provider_call = retry_call
        if not result.content:
            finish_reason = _first_choice_attr(provider_call.response, "finish_reason")
            raise LLMError(f"LLM returned empty content; finish_reason={finish_reason!r}")
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
            if self.fallback_json_actions and self._is_tool_protocol_rejection(exc):
                return await self._complete_json_action_fallback(
                    messages,
                    temperature,
                    max_tokens,
                )
            raise
        result = self._completion_from_chat(provider_call)
        if self._needs_non_thinking_retry(result):
            retry_call = await self._create_chat_completion(
                self._with_enable_thinking(provider_call.request, enabled=False)
            )
            result = self._completion_from_chat(
                retry_call,
                additional_removed=provider_call.compatibility_removed_options,
            )
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
            raise LLMError("LLM returned empty content during JSON action fallback")
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
            # Let the SDK own transient network/rate-limit retry behavior.
            "max_retries": self.max_retries,
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
        instructions, input_items = _messages_to_responses_parts(
            messages,
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
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if self.store:
            payload["store"] = True
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.verbosity:
            payload["verbosity"] = self.verbosity
        self._add_provider_request_options(payload)
        extra_body = self._extra_body()
        if extra_body:
            payload["extra_body"] = extra_body
        return payload

    async def _create_response(self, payload: dict[str, Any]) -> _ProviderCallResult:
        async with self._async_client_scope() as client:
            return await self._call_with_compatibility(client.responses.create, payload, api="responses")

    async def _create_chat_completion(self, payload: dict[str, Any]) -> _ProviderCallResult:
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
        for _attempt in range(self.defaults.compatibility_retry_attempts):
            try:
                response = await create(**request)
                return _ProviderCallResult(
                    response=response,
                    request=dict(request),
                    compatibility_removed_options=tuple(sorted(removed_options)),
                )
            except Exception as exc:
                if not _is_openai_sdk_error(exc):
                    raise
                last_error = exc
                retry = self._compatibility_retry_payload(request, exc, api=api)
                if retry is None:
                    request_id = getattr(exc, "request_id", None)
                    status_code = getattr(exc, "status_code", None)
                    raise LLMError(
                        f"OpenAI SDK {api} request failed: status={status_code!r} request_id={request_id!r} error={exc}"
                    ) from exc
                removed_options.update(
                    key for key in request if key not in retry
                )
                request = retry
        raise LLMError(f"OpenAI SDK {api} request failed after compatibility retries: {last_error}") from last_error

    def _compatibility_retry_payload(self, payload: dict[str, Any], exc: Exception, api: str) -> dict[str, Any] | None:
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
            # GPT-5.6+ uses this newer option instead of retention. Agent
            # libOS does not emit it until it can also model explicit content
            # breakpoints, but compatible callers must still downgrade safely.
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
        if ("text" in message or "json_schema" in message or "json_object" in message) and "text" in retry:
            retry.pop("text", None)
            return retry
        if "response_format" in message and "response_format" in retry:
            retry.pop("response_format", None)
            return retry
        return None

    def _completion_from_response(
        self,
        provider_call: _ProviderCallResult,
        *,
        additional_removed: tuple[str, ...] = (),
    ) -> LLMCompletion:
        response = provider_call.response
        error = getattr(response, "error", None)
        if error is not None:
            raise LLMError(f"OpenAI response failed: {error}")
        tool_calls: list[dict[str, Any]] = []
        for item in getattr(response, "output", None) or []:
            if _get_attr_or_key(item, "type") != "function_call":
                continue
            tool_calls.append(
                {
                    "id": _get_attr_or_key(item, "id") or _get_attr_or_key(item, "call_id"),
                    "call_id": _get_attr_or_key(item, "call_id"),
                    "name": _get_attr_or_key(item, "name"),
                    "arguments": _get_attr_or_key(item, "arguments") or "{}",
                }
            )
        return LLMCompletion(
            content=self._response_text(response),
            tool_calls=tool_calls,
            raw=response,
            api="responses",
            response_id=getattr(response, "id", None),
            request_id=getattr(response, "_request_id", None),
            model=str(getattr(response, "model", "")) or None,
            usage=_usage_from_response(response),
            reasoning=_reasoning_from_response(response),
            provider_request_options=_provider_request_option_observation(
                provider_call.request
            ),
            compatibility_removed_options=sorted(
                set(additional_removed).union(
                    provider_call.compatibility_removed_options
                )
            ),
        )

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
            raise LLMError(f"unexpected LLM response shape: {completion}") from exc

        tool_calls: list[dict[str, Any]] = []
        for call in getattr(message, "tool_calls", None) or []:
            function = getattr(call, "function", None)
            if function is None:
                continue
            tool_calls.append(
                {
                    "id": getattr(call, "id", None),
                    "name": getattr(function, "name", None),
                    "arguments": getattr(function, "arguments", "{}"),
                }
            )

        return LLMCompletion(
            content=self._message_content(message),
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
        )

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
        if self.prompt_cache_retention:
            # Normalize again at the provider boundary so even a caller that
            # mutates the dataclass after construction can never dispatch the
            # retired hyphenated spelling.
            selected_retention = _normalize_prompt_cache_retention(
                self.prompt_cache_retention,
                label="prompt_cache_retention",
            )
            self.prompt_cache_retention = selected_retention
            payload["prompt_cache_retention"] = selected_retention

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
    def _response_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return output_text
        parts: list[str] = []
        for item in getattr(response, "output", None) or []:
            if _get_attr_or_key(item, "type") != "message":
                continue
            for content in _get_attr_or_key(item, "content") or []:
                if _get_attr_or_key(content, "type") == "output_text":
                    parts.append(str(_get_attr_or_key(content, "text") or ""))
        return "".join(parts)

    @staticmethod
    def _message_content(message: Any) -> str:
        content = getattr(message, "content", None)
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(_content_part_text(part) for part in content)
        return str(content)

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
        content = _message_content_for_search(message)
        if role in {"system", "developer"}:
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
                "content": content,
            }
        )
    return ("\n\n".join(instructions) if instructions else None), input_items


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

    return {
        "prompt_cache_key_sent": "prompt_cache_key" in request,
        "prompt_cache_retention": request.get("prompt_cache_retention"),
        "prompt_cache_options_sent": "prompt_cache_options" in request,
        "safety_identifier_sent": "safety_identifier" in request,
    }


def _is_openai_sdk_error(exc: Exception) -> bool:
    try:
        from openai import OpenAIError
    except ImportError:
        return False
    return isinstance(exc, OpenAIError)



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


def _reasoning_from_response(response: Any) -> Any | None:
    direct = _get_attr_or_key(response, "reasoning")
    if direct is not None:
        return to_jsonable(direct)
    reasoning_items: list[Any] = []
    for item in _get_attr_or_key(response, "output") or []:
        if _get_attr_or_key(item, "type") == "reasoning":
            reasoning_items.append(to_jsonable(item))
    return reasoning_items or None


def _reasoning_from_chat_message(message: Any) -> Any | None:
    for key in ("reasoning", "reasoning_content", "thinking", "thinking_content"):
        value = _get_attr_or_key(message, key)
        if _has_value(value):
            return to_jsonable(value)
    additional = _get_attr_or_key(message, "additional_kwargs")
    if isinstance(additional, dict):
        for key in ("reasoning", "reasoning_content", "thinking", "thinking_content"):
            value = additional.get(key)
            if _has_value(value):
                return to_jsonable(value)
    return None


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
