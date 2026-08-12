from __future__ import annotations
import pytest
import asyncio
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any
from pydantic import BaseModel, ConfigDict
from agent_libos.config import AgentLibOSConfig, LLMDefaults
import agent_libos.llm.client as llm_client_module
from agent_libos.llm.client import (
    LLMClient,
    LLMError,
    LLMTransientError,
    LLM_RESPONSE_CONTENT_MAX_CHARS,
    LLM_RESPONSE_TOOL_ARGUMENT_MAX_CHARS,
    LLM_RESPONSE_TOOL_CALL_MAX_COUNT,
    llm_error_internal_observation,
)
from agent_libos.llm.provider_trace import (
    PROVIDER_TRACE_MAX_BYTES,
    ProviderTraceBuilder,
    ProviderTraceAttemptLimitExceeded,
    custom_provider_trace,
    project_provider_raw_response,
    provider_reasoning_view,
    provider_trace_from_error,
)
from agent_libos.utils.serde import dumps
from agent_libos.utils.public_errors import public_error_envelope

class TestLLMClient:

    def test_responses_trace_uses_output_reasoning_not_top_level_configuration(self) -> None:
        class ReasoningItem(BaseModel):
            model_config = ConfigDict(extra="allow")

            type: str
            summary: list[dict[str, str]]
            content: list[dict[str, str]]
            encrypted_content: str | None = None

        secret = "OPAQUE_REASONING_SECRET"
        response = SimpleNamespace(
            id="resp_reasoning",
            model="gpt-test",
            status="completed",
            reasoning={"effort": "high", "summary": "configuration-only"},
            output_text="done",
            output=[
                ReasoningItem(
                    type="reasoning",
                    summary=[{"text": "first summary"}],
                    content=[{"text": "then detail"}],
                    encrypted_content=secret,
                    opaque_blob=secret,
                )
            ],
        )
        fake = FakeAsyncOpenAI(responses=FakeResponses(response))
        client = LLMClient(model="gpt-test", api_key="key", api_mode="responses")
        client._async_client = fake

        completion = asyncio.run(
            client.acomplete_with_metadata(
                messages=[{"role": "user", "content": "answer"}],
                json_mode=False,
            )
        )

        assert completion.provider_trace is not None
        attempt = completion.provider_trace["attempts"][0]
        assert attempt["reasoning"]["blocks"][:2] == [
            {"type": "summary_text", "text": "first summary", "source": "summary"},
            {"type": "reasoning_text", "text": "then detail", "source": "content"},
        ]
        serialized = json.dumps(completion.provider_trace, sort_keys=True)
        assert "configuration-only" not in serialized
        assert secret not in serialized
        assert [block["type"] for block in attempt["reasoning"]["blocks"][2:]] == [
            "opaque",
            "opaque",
        ]

    def test_responses_reasoning_projection_limit_retains_hash_descriptor(self) -> None:
        response = SimpleNamespace(
            id="resp_reasoning_limit",
            model="gpt-test",
            status="completed",
            output_text="done",
            output=[
                {
                    "type": "reasoning",
                    "content": ["x" * 262_144 for _ in range(16)],
                }
            ],
        )
        fake = FakeAsyncOpenAI(responses=FakeResponses(response))
        client = LLMClient(model="gpt-test", api_key="key", api_mode="responses")
        client._async_client = fake

        completion = asyncio.run(
            client.acomplete_with_metadata(
                messages=[{"role": "user", "content": "answer"}],
                json_mode=False,
            )
        )

        assert completion.provider_trace is not None
        reasoning = completion.provider_trace["attempts"][0]["reasoning"]
        assert reasoning["availability"] == "limited"
        omitted = next(
            block for block in reasoning["blocks"] if block["type"] == "omitted"
        )
        assert omitted["reason"] in {"aggregate_limit", "bounds"}
        assert omitted["bytes"] > 0
        assert len(omitted["sha256"]) == 64

    def test_responses_absent_opaque_reasoning_fields_do_not_claim_content(self) -> None:
        response = SimpleNamespace(
            id="resp_reasoning_optional_none",
            model="gpt-test",
            status="completed",
            output_text="done",
            output=[
                SimpleNamespace(
                    type="reasoning",
                    summary=None,
                    content=None,
                    encrypted_content=None,
                    signature=None,
                )
            ],
        )
        fake = FakeAsyncOpenAI(responses=FakeResponses(response))
        client = LLMClient(model="gpt-test", api_key="key", api_mode="responses")
        client._async_client = fake

        completion = asyncio.run(
            client.acomplete_with_metadata(
                messages=[{"role": "user", "content": "answer"}],
                json_mode=False,
            )
        )

        assert completion.provider_trace is not None
        reasoning = completion.provider_trace["attempts"][0]["reasoning"]
        assert reasoning == {"availability": "not_returned", "blocks": []}

    def test_explicit_transport_retry_records_each_wire_attempt(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class RateLimited(Exception):
            status_code = 429
            response = SimpleNamespace(headers={"retry-after": "0"})

        completion = SimpleNamespace(
            id="chat_retry",
            model="gpt-test",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="ok", tool_calls=[]),
                )
            ],
        )
        fake = FakeAsyncOpenAI(
            chat=FakeChat(FakeChatCompletions([RateLimited("private"), completion]))
        )
        client = LLMClient(
            model="gpt-test",
            api_key="key",
            api_mode="chat",
            max_retries=2,
        )
        client._async_client = fake
        monkeypatch.setattr(llm_client_module, "_is_openai_sdk_error", lambda _exc: True)

        result = asyncio.run(
            client.acomplete_with_metadata(
                messages=[{"role": "user", "content": "answer"}],
                json_mode=False,
            )
        )

        assert result.provider_trace is not None
        assert result.provider_trace["selected_attempt"] == 2
        assert [attempt["kind"] for attempt in result.provider_trace["attempts"]] == [
            "initial",
            "transport_retry",
        ]
        assert [attempt["status"] for attempt in result.provider_trace["attempts"]] == [
            "error",
            "ok",
        ]
        assert result.provider_trace["attempts"][0]["error"]["status_code"] == 429
        assert len(fake.chat.completions.payloads) == 2

    def test_compatibility_retry_is_distinct_from_transport_retry(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class ProviderError(Exception):
            def __init__(self, message: str, status_code: int) -> None:
                super().__init__(message)
                self.status_code = status_code
                self.response = SimpleNamespace(headers={"retry-after": "0"})

        completion = SimpleNamespace(
            id="chat_compat",
            model="gpt-test",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="ok", tool_calls=[]),
                )
            ],
        )
        fake = FakeAsyncOpenAI(
            chat=FakeChat(
                FakeChatCompletions(
                    [
                        ProviderError("rate limited", 429),
                        ProviderError("unknown parameter temperature", 400),
                        completion,
                    ]
                )
            )
        )
        client = LLMClient(
            model="gpt-test",
            api_key="key",
            api_mode="chat",
            max_retries=1,
        )
        client._async_client = fake
        monkeypatch.setattr(llm_client_module, "_is_openai_sdk_error", lambda _exc: True)

        result = asyncio.run(
            client.acomplete_with_metadata(
                messages=[{"role": "user", "content": "answer"}],
                json_mode=False,
            )
        )

        assert result.provider_trace is not None
        assert [attempt["kind"] for attempt in result.provider_trace["attempts"]] == [
            "initial",
            "transport_retry",
            "compatibility_retry",
        ]
        assert "temperature" not in fake.chat.completions.payloads[2]

    def test_non_thinking_retry_rejects_first_attempt_and_selects_second(self) -> None:
        first = SimpleNamespace(
            id="chat_empty",
            model="compat-model",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="", tool_calls=[]),
                )
            ],
        )
        second = SimpleNamespace(
            id="chat_final",
            model="compat-model",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="ok", tool_calls=[]),
                )
            ],
        )
        fake = FakeAsyncOpenAI(
            chat=FakeChat(FakeChatCompletions([first, second]))
        )
        client = LLMClient(
            base_url="https://example.com/v1",
            model="compat-model",
            api_key="key",
            api_mode="chat",
            allow_custom_base_url=True,
        )
        client._async_client = fake

        result = asyncio.run(
            client.acomplete_with_metadata(
                messages=[{"role": "user", "content": "answer"}],
                json_mode=False,
            )
        )

        assert result.provider_trace is not None
        assert result.provider_trace["selected_attempt"] == 2
        assert [attempt["kind"] for attempt in result.provider_trace["attempts"]] == [
            "initial",
            "non_thinking_retry",
        ]
        assert [attempt["status"] for attempt in result.provider_trace["attempts"]] == [
            "error",
            "ok",
        ]

    def test_terminal_provider_error_carries_bounded_trace(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class BadRequest(Exception):
            status_code = 400
            response = SimpleNamespace(headers={"x-should-retry": "false"})

        secret = "PRIVATE_PROVIDER_ERROR"
        fake = FakeAsyncOpenAI(
            chat=FakeChat(FakeChatCompletions(BadRequest(secret)))
        )
        client = LLMClient(
            model="gpt-test",
            api_key="key",
            api_mode="chat",
            max_retries=3,
        )
        client._async_client = fake
        monkeypatch.setattr(llm_client_module, "_is_openai_sdk_error", lambda _exc: True)

        with pytest.raises(LLMError) as raised:
            asyncio.run(
                client.acomplete_with_metadata(
                    messages=[{"role": "user", "content": "answer"}],
                    json_mode=False,
                )
            )

        trace = provider_trace_from_error(raised.value)
        assert trace is not None
        assert len(trace["attempts"]) == 1
        assert trace["attempts"][0]["status"] == "error"
        assert secret not in json.dumps(trace, sort_keys=True)
        assert len(fake.chat.completions.payloads) == 1

    def test_redirect_is_terminal_even_when_provider_requests_retry(self) -> None:
        error = SimpleNamespace(
            status_code=302,
            response=SimpleNamespace(headers={"x-should-retry": "true"}),
        )

        assert llm_client_module._should_retry_openai_sdk_error(error) is False

    @pytest.mark.parametrize(
        ("status_code", "headers", "expected"),
        [
            (408, {}, True),
            (409, {}, True),
            (429, {}, True),
            (500, {}, True),
            (599, {}, True),
            (400, {}, False),
            (400, {"x-should-retry": "true"}, True),
            (503, {"x-should-retry": "false"}, False),
        ],
    )
    def test_explicit_retry_status_and_header_matrix(
        self,
        status_code: int,
        headers: dict[str, str],
        expected: bool,
    ) -> None:
        error = SimpleNamespace(
            status_code=status_code,
            response=SimpleNamespace(headers=headers),
        )

        assert llm_client_module._should_retry_openai_sdk_error(error) is expected

    @pytest.mark.parametrize("error_type", ["APIConnectionError", "APITimeoutError"])
    def test_explicit_retry_transport_error_matrix(self, error_type: str) -> None:
        error = type(error_type, (Exception,), {})("private transport failure")

        assert llm_client_module._should_retry_openai_sdk_error(error) is True

    @pytest.mark.parametrize(
        ("headers", "retry_index", "expected"),
        [
            ({"retry-after-ms": "1500"}, 0, 1.5),
            ({"retry-after": "60"}, 0, 60.0),
            ({"retry-after": "61"}, 2, 2.0),
            ({"retry-after": "invalid"}, 2, 2.0),
        ],
    )
    def test_retry_after_bounds_and_exponential_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        headers: dict[str, str],
        retry_index: int,
        expected: float,
    ) -> None:
        monkeypatch.setattr(llm_client_module.random, "random", lambda: 0.5)
        error = SimpleNamespace(response=SimpleNamespace(headers=headers))

        assert llm_client_module._openai_retry_delay(error, retry_index) == expected

    def test_custom_provider_error_does_not_claim_an_unobservable_wire_attempt(self) -> None:
        trace = custom_provider_trace(error=RuntimeError("private adapter error"))

        assert trace["coverage"] == "custom_client_incomplete"
        assert trace["selected_attempt"] is None
        assert trace["attempts"] == []

    def test_raw_response_projection_hashes_opaque_fields_and_caps_aggregate(self) -> None:
        secret = "ENCRYPTED_PROVIDER_SECRET"
        projected = project_provider_raw_response(
            {
                "encrypted_content": secret,
                "signature": secret,
                "opaque_data": secret,
                "opaque_wrapper": {"type": "opaque", "data": secret},
                ("a" * 256) + "_encrypted_content": secret,
                "parts": ["x" * 200_000 for _ in range(30)],
            }
        )
        serialized = dumps(projected).encode()
        assert len(serialized) <= PROVIDER_TRACE_MAX_BYTES
        assert secret.encode() not in serialized
        assert projected["encrypted_content"]["sha256"]
        assert projected["opaque_data"]["type"] == "opaque"
        assert projected["opaque_wrapper"]["type"] == "opaque"
        assert projected["a" * 256]["type"] == "opaque"

        credential_secret = "RAW_PROVIDER_CREDENTIAL_SECRET"
        credential_projection = project_provider_raw_response(
            {
                "Authorization": f"Bearer {credential_secret}",
                "headers": [
                    ["x-api-key", credential_secret],
                    ["Authorization", credential_secret, credential_secret],
                    ["content-type", "application/json"],
                    {"name": "Authorization", "value": credential_secret},
                    {"key": "set-cookie", "data": [credential_secret]},
                    {"name": "content-type", "value": "application/json"},
                ],
            }
        )
        credential_wire = dumps(credential_projection)
        assert credential_secret not in credential_wire
        assert credential_projection["Authorization"]["type"] == "opaque"
        assert credential_projection["headers"][0][0] == "x-api-key"
        assert credential_projection["headers"][0][1]["type"] == "opaque"
        assert credential_projection["headers"][1][1]["type"] == "opaque"
        assert credential_projection["headers"][1][2]["type"] == "opaque"
        assert credential_projection["headers"][2] == [
            "content-type",
            "application/json",
        ]
        assert credential_projection["headers"][3]["value"]["type"] == "opaque"
        assert credential_projection["headers"][4]["data"]["type"] == "opaque"
        assert credential_projection["headers"][5] == {
            "name": "content-type",
            "value": "application/json",
        }

        opaque_reasoning = provider_reasoning_view({"opaque": secret})
        assert opaque_reasoning["availability"] == "returned"
        assert opaque_reasoning["blocks"][0]["type"] == "opaque"
        assert secret not in dumps(opaque_reasoning)

        nested: dict[str, Any] = {"text": "leaf"}
        for _ in range(40):
            nested = {"content": nested}
        reasoning = provider_reasoning_view(nested)
        assert reasoning["availability"] == "limited"
        omitted = next(
            block for block in reasoning["blocks"] if block["type"] == "omitted"
        )
        assert omitted["bytes"] > 0
        assert len(omitted["sha256"]) == 64

        many_blocks = provider_reasoning_view(["x"] * 10_000)
        assert many_blocks["availability"] == "limited"
        assert len(many_blocks["blocks"]) <= 4_096
        assert many_blocks["blocks"][-1]["type"] == "omitted"

        huge_integer = 10**5_000
        projected_integer = project_provider_raw_response({"value": huge_integer})
        assert projected_integer["value"]["reason"] == "integer_out_of_range"
        assert projected_integer["value"]["sha256"]

        builder = ProviderTraceBuilder()
        sequence = builder.start_attempt(api="chat", kind="initial")
        builder.finish_response(
            sequence,
            SimpleNamespace(usage={"total_tokens": huge_integer}),
        )
        huge_integer_trace = builder.to_dict()
        usage = huge_integer_trace["attempts"][0]["usage"]
        assert usage["total_tokens"] is None
        assert usage["_provider_projection_invalid_fields"] == ["total_tokens"]

        class WideSdkModel:
            def __init__(self) -> None:
                self.__dict__.update({f"field_{index}": index for index in range(10_000)})

            def model_dump(self) -> dict[str, Any]:
                raise AssertionError("unbounded recursive model_dump must not be called")

        projected_model = project_provider_raw_response(WideSdkModel())
        assert projected_model["_provider_projection_limited"] is True
        assert projected_model["_omitted"]["items"] > 0
        assert len(projected_model) <= 4_097

    def test_attempt_limit_stops_the_257th_dispatch_before_provider(self) -> None:
        builder = ProviderTraceBuilder()
        for _ in range(256):
            builder.start_attempt(api="chat", kind="initial")
        dispatched = 0

        async def create(**_payload: Any) -> Any:
            nonlocal dispatched
            dispatched += 1
            return SimpleNamespace()

        async def run() -> None:
            token = llm_client_module._ACTIVE_PROVIDER_TRACE.set(builder)
            try:
                client = LLMClient(
                    model="gpt-test",
                    api_key="key",
                    api_mode="chat",
                    max_retries=0,
                )
                await client._call_with_transport_retries(
                    create,
                    {"model": "gpt-test"},
                    api="chat",
                    kind="initial",
                )
            finally:
                llm_client_module._ACTIVE_PROVIDER_TRACE.reset(token)

        with pytest.raises(ProviderTraceAttemptLimitExceeded):
            asyncio.run(run())
        assert dispatched == 0

    def test_provider_trace_cap_matches_durable_serializer(self) -> None:
        builder = ProviderTraceBuilder()
        response = SimpleNamespace(id="response", model="model", usage={})
        for _ in range(256):
            sequence = builder.start_attempt(api="chat", kind="initial")
            builder.finish_response(sequence, response)
            builder.enrich_response(
                sequence,
                reasoning=None,
                output="x" * 16_100,
                tool_calls=[],
                usage={},
            )
        builder.mark_selected(256)

        trace = builder.to_dict()

        assert len(dumps(trace).encode()) <= PROVIDER_TRACE_MAX_BYTES
        assert trace["selected_attempt"] == 256

    def test_sdk_timeout_is_classified_as_transient_after_sdk_retries(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class APITimeoutError(Exception):
            pass

        async def timeout(**_payload: object) -> object:
            raise APITimeoutError("Request timed out.")

        client = LLMClient(model="gpt-test", api_key="key", api_mode="chat")
        monkeypatch.setattr(llm_client_module, "_is_openai_sdk_error", lambda _exc: True)

        with pytest.raises(LLMTransientError) as raised:
            asyncio.run(
                client._call_with_compatibility(
                    timeout,
                    {"model": "gpt-test"},
                    api="chat",
                )
            )
        public_error = public_error_envelope(raised.value)
        observation = llm_error_internal_observation(
            raised.value,
            correlation_id=public_error["correlation_id"],
        )
        assert str(raised.value) == public_error["message"]
        assert "Request timed out" not in str(raised.value)
        assert observation["correlation_id"] == public_error["correlation_id"]
        assert observation["exception_text"] == {
            "bytes": len("Request timed out.".encode()),
            "sha256": hashlib.sha256("Request timed out.".encode()).hexdigest(),
        }
        assert isinstance(raised.value.__cause__, APITimeoutError)

    def test_nonretryable_sdk_status_remains_terminal_llm_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class BadRequestError(Exception):
            status_code = 400

        async def reject(**_payload: object) -> object:
            raise BadRequestError("invalid request")

        client = LLMClient(model="gpt-test", api_key="key", api_mode="chat")
        monkeypatch.setattr(llm_client_module, "_is_openai_sdk_error", lambda _exc: True)

        with pytest.raises(LLMError) as raised:
            asyncio.run(
                client._call_with_compatibility(
                    reject,
                    {"model": "gpt-test"},
                    api="chat",
                )
            )

        assert not isinstance(raised.value, LLMTransientError)
        assert "invalid request" not in str(raised.value)
        assert str(raised.value).startswith("llm_error: LLMError (correlation_id=")

    def test_real_async_client_is_not_reused_across_closed_event_loops(self) -> None:
        class PerTestKeepAliveChatHandler(_KeepAliveChatHandler):
            request_count = 0

        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            PerTestKeepAliveChatHandler,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = LLMClient(
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="gpt-test",
            api_key="key",
            api_mode="chat",
            allow_custom_base_url=True,
        )
        try:
            messages = [{"role": "user", "content": "reply"}]

            first = asyncio.run(client.acomplete(messages, json_mode=False))
            second = asyncio.run(client.acomplete(messages, json_mode=False))

            assert first == "ok"
            assert second == "ok"
            assert PerTestKeepAliveChatHandler.request_count == 2
        finally:
            client.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=1.0)

    def test_responses_action_request_converts_chat_tools_and_parses_function_calls(self) -> None:
        response = SimpleNamespace(id='resp_123', model='gpt-test', usage=SimpleNamespace(input_tokens=11, output_tokens=3, total_tokens=14), output_text='', output=[SimpleNamespace(type='reasoning', summary=[{'text': 'choose write_text_file'}]), SimpleNamespace(type='function_call', id='fc_1', call_id='call_1', name='write_text_file', arguments='{"path":"out.txt","content":"ok"}')])
        fake = FakeAsyncOpenAI(responses=FakeResponses(response))
        client = LLMClient(model='gpt-test', api_key='key', api_mode='responses')
        client._async_client = fake
        assert client.store is False
        assert client.responses_previous_response_id is False
        completion = asyncio.run(client.acomplete_action(messages=[{'role': 'system', 'content': 'system rules'}, {'role': 'user', 'content': 'write a file'}], tools=[{'type': 'function', 'function': {'name': 'write_text_file', 'description': 'Write text.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}}}}]))
        payload = fake.responses.payloads[0]
        assert payload['instructions'] == 'system rules'
        assert payload['input'] == [{'role': 'user', 'content': 'write a file'}]
        assert payload['tools'][0]['name'] == 'write_text_file'
        assert payload['tools'][0]['type'] == 'function'
        assert payload['tools'][0]['strict']
        assert payload['tools'][0]['parameters']['additionalProperties'] is False
        assert payload['tools'][0]['parameters']['required'] == ['path']
        assert not payload['parallel_tool_calls']
        assert not payload['store']
        assert completion.api == 'responses'
        assert completion.response_id == 'resp_123'
        assert completion.tool_calls[0]['call_id'] == 'call_1'
        assert completion.tool_calls[0]['name'] == 'write_text_file'
        assert completion.usage['total_tokens'] == 14
        assert completion.reasoning[0]['summary'][0]['text'] == 'choose write_text_file'

    def test_responses_action_rejects_incomplete_function_call(self) -> None:
        response = SimpleNamespace(
            id="resp_incomplete",
            model="gpt-test",
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            usage=SimpleNamespace(input_tokens=9, output_tokens=2, total_tokens=11),
            output_text="",
            output=[
                SimpleNamespace(
                    type="function_call",
                    status="incomplete",
                    id="fc_incomplete",
                    call_id="call_incomplete",
                    name="process_exit",
                    arguments="{}",
                )
            ],
        )
        fake = FakeAsyncOpenAI(responses=FakeResponses(response))
        client = LLMClient(model="gpt-test", api_key="key", api_mode="responses")
        client._async_client = fake

        with pytest.raises(LLMError) as raised:
            asyncio.run(
                client.acomplete_action(
                    messages=[{"role": "user", "content": "exit"}],
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "process_exit",
                                "description": "Exit.",
                                "parameters": {"type": "object", "properties": {}},
                            },
                        }
                    ],
                )
            )
        assert "incomplete" not in str(raised.value)
        assert str(raised.value).startswith("llm_error: LLMError (correlation_id=")
        trace = provider_trace_from_error(raised.value)
        assert trace is not None
        assert trace["selected_attempt"] is None
        assert trace["attempts"][0]["status"] == "error"
        assert trace["attempts"][0]["output"] == ""
        assert trace["attempts"][0]["usage"] == {
            "input_tokens": 9,
            "output_tokens": 2,
            "total_tokens": 11,
        }

    def test_responses_provider_error_object_is_text_free(self) -> None:
        secret = "PROVIDER_RESPONSE_ERROR_SECRET"
        response = SimpleNamespace(error=secret)
        fake = FakeAsyncOpenAI(responses=FakeResponses(response))
        client = LLMClient(model="gpt-test", api_key="key", api_mode="responses")
        client._async_client = fake

        with pytest.raises(LLMError) as raised:
            asyncio.run(
                client.acomplete_action(
                    messages=[{"role": "user", "content": "exit"}],
                    tools=[],
                )
            )

        public_error = public_error_envelope(raised.value)
        observation = llm_error_internal_observation(
            raised.value,
            correlation_id=public_error["correlation_id"],
        )
        assert str(raised.value) == public_error["message"]
        assert secret not in str(raised.value)
        assert observation["exception_text"] == {
            "bytes": len(secret.encode()),
            "sha256": hashlib.sha256(secret.encode()).hexdigest(),
        }

    def test_unexpected_chat_shape_does_not_repr_provider_response(self) -> None:
        secret = "PROVIDER_RESPONSE_REPR_SECRET"

        class UnexpectedResponse:
            choices: list[object] = []

            def __repr__(self) -> str:
                return secret

        fake = FakeAsyncOpenAI(
            chat=FakeChat(FakeChatCompletions(UnexpectedResponse()))
        )
        client = LLMClient(
            base_url="https://example.com/compatible/v1",
            model="compat-model",
            api_key="key",
            api_mode="chat",
            allow_custom_base_url=True,
        )
        client._async_client = fake

        with pytest.raises(LLMError) as raised:
            asyncio.run(
                client.acomplete_action(
                    messages=[{"role": "user", "content": "exit"}],
                    tools=[],
                )
            )

        assert secret not in str(raised.value)
        assert str(raised.value).startswith("llm_error: LLMError (correlation_id=")

    @pytest.mark.parametrize("api", ["responses", "chat"])
    def test_provider_response_rejects_oversized_content(self, api: str) -> None:
        content = "x" * (LLM_RESPONSE_CONTENT_MAX_CHARS + 1)
        if api == "responses":
            response = SimpleNamespace(
                id="resp_oversized",
                model="gpt-test",
                status="completed",
                output_text=content,
                output=[],
            )
            fake = FakeAsyncOpenAI(responses=FakeResponses(response))
        else:
            response = SimpleNamespace(
                id="chat_oversized",
                model="gpt-test",
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content=content, tool_calls=[]),
                    )
                ],
            )
            fake = FakeAsyncOpenAI(chat=FakeChat(FakeChatCompletions(response)))
        client = LLMClient(model="gpt-test", api_key="key", api_mode=api)
        client._async_client = fake

        with pytest.raises(LLMError) as raised:
            asyncio.run(client.acomplete_action(messages=[], tools=[]))

        assert content[:100] not in str(raised.value)
        assert str(raised.value).startswith("llm_error: LLMError (correlation_id=")

    @pytest.mark.parametrize("api", ["responses", "chat"])
    def test_provider_response_rejects_excess_tool_calls(self, api: str) -> None:
        if api == "responses":
            output = [
                SimpleNamespace(
                    type="function_call",
                    status="completed",
                    id=f"fc_{index}",
                    call_id=f"call_{index}",
                    name="process_exit",
                    arguments="{}",
                )
                for index in range(LLM_RESPONSE_TOOL_CALL_MAX_COUNT + 1)
            ]
            response = SimpleNamespace(
                id="resp_many_tools",
                model="gpt-test",
                status="completed",
                output_text="",
                output=output,
            )
            fake = FakeAsyncOpenAI(responses=FakeResponses(response))
        else:
            calls = [
                SimpleNamespace(
                    id=f"call_{index}",
                    function=SimpleNamespace(
                        name="process_exit",
                        arguments="{}",
                    ),
                )
                for index in range(LLM_RESPONSE_TOOL_CALL_MAX_COUNT + 1)
            ]
            response = SimpleNamespace(
                id="chat_many_tools",
                model="gpt-test",
                choices=[
                    SimpleNamespace(
                        finish_reason="tool_calls",
                        message=SimpleNamespace(content="", tool_calls=calls),
                    )
                ],
            )
            fake = FakeAsyncOpenAI(chat=FakeChat(FakeChatCompletions(response)))
        client = LLMClient(model="gpt-test", api_key="key", api_mode=api)
        client._async_client = fake

        with pytest.raises(LLMError):
            asyncio.run(client.acomplete_action(messages=[], tools=[]))

    @pytest.mark.parametrize("api", ["responses", "chat"])
    def test_provider_response_rejects_oversized_tool_arguments(self, api: str) -> None:
        arguments = "x" * (LLM_RESPONSE_TOOL_ARGUMENT_MAX_CHARS + 1)
        if api == "responses":
            output = [
                SimpleNamespace(
                    type="function_call",
                    status="completed",
                    id="fc_1",
                    call_id="call_1",
                    name="process_exit",
                    arguments=arguments,
                )
            ]
            response = SimpleNamespace(
                id="resp_large_arguments",
                model="gpt-test",
                status="completed",
                output_text="",
                output=output,
            )
            fake = FakeAsyncOpenAI(responses=FakeResponses(response))
        else:
            response = SimpleNamespace(
                id="chat_large_arguments",
                model="gpt-test",
                choices=[
                    SimpleNamespace(
                        finish_reason="tool_calls",
                        message=SimpleNamespace(
                            content="",
                            tool_calls=[
                                SimpleNamespace(
                                    id="call_1",
                                    function=SimpleNamespace(
                                        name="process_exit",
                                        arguments=arguments,
                                    ),
                                )
                            ],
                        ),
                    )
                ],
            )
            fake = FakeAsyncOpenAI(chat=FakeChat(FakeChatCompletions(response)))
        client = LLMClient(model="gpt-test", api_key="key", api_mode=api)
        client._async_client = fake

        with pytest.raises(LLMError):
            asyncio.run(client.acomplete_action(messages=[], tools=[]))

    def test_responses_action_request_preserves_native_function_transcript(self) -> None:
        response = SimpleNamespace(
            id="resp_transcript_2",
            model="gpt-test",
            usage=SimpleNamespace(input_tokens=11, output_tokens=3, total_tokens=14),
            output_text="",
            output=[],
        )
        fake = FakeAsyncOpenAI(responses=FakeResponses(response))
        client = LLMClient(model="gpt-test", api_key="key", api_mode="responses")
        client._async_client = fake

        asyncio.run(
            client.acomplete_action(
                messages=[
                    {"role": "system", "content": "exact system"},
                    {"role": "user", "content": "call the tool"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_transcript_1",
                                "type": "function",
                                "function": {
                                    "name": "echo",
                                    "arguments": '{"value":"done"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_transcript_1",
                        "name": "echo",
                        "content": '{"value":"done"}',
                    },
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "process_exit",
                            "description": "Exit.",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )
        )

        assert fake.responses.payloads[0]["instructions"] == "exact system"
        assert "previous_response_id" not in fake.responses.payloads[0]
        assert fake.responses.payloads[0]["input"] == [
            {"role": "user", "content": "call the tool"},
            {
                "type": "function_call",
                "call_id": "call_transcript_1",
                "name": "echo",
                "arguments": '{"value":"done"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_transcript_1",
                "output": '{"value":"done"}',
            },
        ]

    def test_action_requests_send_configured_parallel_tool_calls(self) -> None:
        response = SimpleNamespace(id='resp_parallel', model='gpt-test', output_text='', output=[])
        fake = FakeAsyncOpenAI(responses=FakeResponses(response))
        client = LLMClient(model='gpt-test', api_key='key', api_mode='responses', parallel_tool_calls=True)
        client._async_client = fake

        asyncio.run(
            client.acomplete_action(
                messages=[{'role': 'user', 'content': 'call tools'}],
                tools=[{'type': 'function', 'function': {'name': 'process_exit', 'description': 'Exit.', 'parameters': {'type': 'object', 'properties': {}}}}],
            )
        )

        assert fake.responses.payloads[0]['parallel_tool_calls'] is True

        chat_completion = SimpleNamespace(
            id='chatcmpl_parallel',
            model='gpt-test',
            choices=[SimpleNamespace(finish_reason='tool_calls', message=SimpleNamespace(content='', tool_calls=[]))],
        )
        chat_fake = FakeAsyncOpenAI(chat=FakeChat(FakeChatCompletions(chat_completion)))
        chat_client = LLMClient(model='gpt-test', api_key='key', api_mode='chat', parallel_tool_calls=True)
        chat_client._async_client = chat_fake

        asyncio.run(
            chat_client.acomplete_action(
                messages=[{'role': 'user', 'content': 'call tools'}],
                tools=[{'type': 'function', 'function': {'name': 'process_exit', 'description': 'Exit.', 'parameters': {'type': 'object', 'properties': {}}}}],
            )
        )

        assert chat_fake.chat.completions.payloads[0]['parallel_tool_calls'] is True

    def test_responses_text_request_uses_json_mode_when_requested(self) -> None:
        response = SimpleNamespace(id='resp_json', model='gpt-test', output_text='{"ok":true}', output=[])
        fake = FakeAsyncOpenAI(responses=FakeResponses(response))
        client = LLMClient(model='gpt-test', api_key='key', api_mode='responses', verbosity='low')
        client._async_client = fake
        content = asyncio.run(client.acomplete([{'role': 'user', 'content': 'return json'}], json_mode=True))
        payload = fake.responses.payloads[0]
        assert content == '{"ok":true}'
        assert payload['text']['format'] == {'type': 'json_object'}
        assert payload['text']['verbosity'] == 'low'

    def test_responses_tool_schema_keeps_dynamic_objects_non_strict(self) -> None:
        response = SimpleNamespace(id='resp_dynamic', model='gpt-test', output_text='', output=[])
        fake = FakeAsyncOpenAI(responses=FakeResponses(response))
        client = LLMClient(model='gpt-test', api_key='key', api_mode='responses')
        client._async_client = fake

        asyncio.run(
            client.acomplete_action(
                messages=[{'role': 'user', 'content': 'call dynamic'}],
                tools=[
                    {
                        'type': 'function',
                        'function': {
                            'name': 'dynamic_tool',
                            'description': 'Accept dynamic args.',
                            'parameters': {'type': 'object', 'additionalProperties': True},
                        },
                    }
                ],
            )
        )

        tool = fake.responses.payloads[0]['tools'][0]
        assert tool['strict'] is False
        assert tool['parameters'] == {'type': 'object', 'additionalProperties': True}

    def test_responses_text_request_uses_json_schema_when_provided(self) -> None:
        response = SimpleNamespace(id='resp_schema', model='gpt-test', output_text='{"ok":true}', output=[])
        fake = FakeAsyncOpenAI(responses=FakeResponses(response))
        client = LLMClient(model='gpt-test', api_key='key', api_mode='responses', verbosity='low')
        client._async_client = fake

        content = asyncio.run(
            client.acomplete(
                [{'role': 'user', 'content': 'return json'}],
                json_schema={'type': 'object', 'properties': {'ok': {'type': 'boolean'}}},
                schema_name='test_response',
            )
        )

        payload = fake.responses.payloads[0]
        assert content == '{"ok":true}'
        assert payload['text']['verbosity'] == 'low'
        assert payload['text']['format']['type'] == 'json_schema'
        assert payload['text']['format']['name'] == 'test_response'
        assert payload['text']['format']['strict'] is True
        assert payload['text']['format']['schema']['additionalProperties'] is False
        assert payload['text']['format']['schema']['required'] == ['ok']

    def test_chat_text_request_uses_json_schema_when_provided(self) -> None:
        chat_completion = SimpleNamespace(
            id='chatcmpl_schema',
            model='gpt-test',
            choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content='{"ok":true}', tool_calls=[]))],
        )
        fake = FakeAsyncOpenAI(chat=FakeChat(FakeChatCompletions(chat_completion)))
        client = LLMClient(model='gpt-test', api_key='key', api_mode='chat')
        client._async_client = fake

        asyncio.run(
            client.acomplete(
                [{'role': 'user', 'content': 'return json'}],
                json_schema={'type': 'object', 'properties': {'ok': {'type': 'boolean'}}},
                schema_name='chat_response',
            )
        )

        response_format = fake.chat.completions.payloads[0]['response_format']
        assert response_format['type'] == 'json_schema'
        assert response_format['json_schema']['name'] == 'chat_response'
        assert response_format['json_schema']['strict'] is True
        assert response_format['json_schema']['schema']['required'] == ['ok']

    def test_explicit_responses_options_are_sent_to_configured_provider(self) -> None:
        response = SimpleNamespace(id='resp_options', model='gpt-test', output_text='', output=[])
        fake = FakeAsyncOpenAI(responses=FakeResponses(response))
        client = LLMClient(
            model='gpt-test',
            api_key='key',
            api_mode='responses',
            store=True,
            safety_identifier='session-safe',
            prompt_cache_key='cache-key',
            prompt_cache_retention='in-memory',
        )
        client._async_client = fake

        asyncio.run(
            client.acomplete_action(
                messages=[{'role': 'user', 'content': 'exit'}],
                tools=[{'type': 'function', 'function': {'name': 'process_exit', 'description': 'Exit.', 'parameters': {'type': 'object', 'properties': {}}}}],
                previous_response_id='resp_prev',
            )
        )

        payload = fake.responses.payloads[0]
        assert payload['previous_response_id'] == 'resp_prev'
        assert payload['safety_identifier'] == 'session-safe'
        assert payload['prompt_cache_key'] == 'cache-key'
        assert client.prompt_cache_retention == 'in_memory'
        assert payload['prompt_cache_retention'] == 'in_memory'

        no_store_fake = FakeAsyncOpenAI(responses=FakeResponses(response))
        no_store = LLMClient(model='gpt-test', api_key='key', api_mode='responses', store=False)
        no_store._async_client = no_store_fake
        asyncio.run(
            no_store.acomplete_action(
                messages=[{'role': 'user', 'content': 'exit'}],
                tools=[{'type': 'function', 'function': {'name': 'process_exit', 'description': 'Exit.', 'parameters': {'type': 'object', 'properties': {}}}}],
                previous_response_id='resp_prev',
            )
        )
        assert 'previous_response_id' not in no_store_fake.responses.payloads[0]

        custom_fake = FakeAsyncOpenAI(responses=FakeResponses(response))
        custom = LLMClient(
            base_url='https://example.com/compatible/v1',
            model='compat-model',
            api_key='key',
            api_mode='responses',
            allow_custom_base_url=True,
            store=True,
            safety_identifier='session-safe',
            prompt_cache_key='cache-key',
            prompt_cache_retention='24h',
        )
        custom._async_client = custom_fake
        asyncio.run(
            custom.acomplete_action(
                messages=[{'role': 'user', 'content': 'exit'}],
                tools=[{'type': 'function', 'function': {'name': 'process_exit', 'description': 'Exit.', 'parameters': {'type': 'object', 'properties': {}}}}],
                previous_response_id='resp_prev',
            )
        )
        custom_payload = custom_fake.responses.payloads[0]
        assert 'previous_response_id' not in custom_payload
        assert custom_payload['safety_identifier'] == 'session-safe'
        assert custom_payload['prompt_cache_key'] == 'cache-key'
        assert custom_payload['prompt_cache_retention'] == '24h'

    def test_chat_sends_explicit_cache_and_safety_options(self) -> None:
        chat_completion = SimpleNamespace(
            id='chatcmpl_options',
            model='gpt-test',
            choices=[
                SimpleNamespace(
                    finish_reason='stop',
                    message=SimpleNamespace(content='ok', tool_calls=[]),
                )
            ],
        )
        fake = FakeAsyncOpenAI(chat=FakeChat(FakeChatCompletions(chat_completion)))
        client = LLMClient(
            base_url='https://example.com/compatible/v1',
            model='compat-model',
            api_key='key',
            api_mode='chat',
            allow_custom_base_url=True,
            safety_identifier='session-safe',
            prompt_cache_key='cache-key',
            prompt_cache_retention='24h',
        )
        client._async_client = fake

        completion = asyncio.run(
            client.acomplete(
                [{'role': 'user', 'content': 'answer'}],
                json_mode=False,
            )
        )

        payload = fake.chat.completions.payloads[0]
        assert payload['safety_identifier'] == 'session-safe'
        assert payload['prompt_cache_key'] == 'cache-key'
        assert payload['prompt_cache_retention'] == '24h'
        assert completion == 'ok'

    @pytest.mark.parametrize("api", ["chat", "responses"])
    def test_v2_explicit_cache_marks_only_the_host_stable_text_prefix(
        self,
        api: str,
    ) -> None:
        client = LLMClient(
            model="gpt-5.6",
            api_key="key",
            api_mode=api,
            prompt_cache_key="tenant-domain",
            prompt_cache_mode="explicit",
            prompt_cache_ttl="30m",
        )
        messages = [
            {"role": "system", "content": "stable system"},
            {
                "role": "user",
                "content": "stable task contract",
                "_agent_libos_cache_stable": True,
            },
            {"role": "user", "content": "dynamic tail"},
        ]

        payload = (
            client._chat_payload(messages, 0.0, 100)
            if api == "chat"
            else client._responses_payload(messages, 0.0, 100)
        )
        payload["tools"] = []
        client._finalize_prompt_cache_request(payload)
        encoded = json.dumps(payload, sort_keys=True)

        assert payload["prompt_cache_options"] == {
            "mode": "explicit",
            "ttl": "30m",
        }
        assert payload["prompt_cache_key"].startswith("alibos:v2:")
        assert "tenant-domain" not in payload["prompt_cache_key"]
        assert encoded.count("prompt_cache_breakpoint") == 1
        assert "_agent_libos_cache_stable" not in encoded
        assert "stable task contract" in encoded
        assert "dynamic tail" in encoded

    def test_v2_implicit_cache_uses_stable_fingerprint_without_wire_breakpoint(self) -> None:
        client = LLMClient(
            model="gpt-5.6",
            api_key="key",
            api_mode="chat",
            prompt_cache_key="tenant-domain",
            prompt_cache_mode="implicit",
        )
        messages = [
            {"role": "system", "content": "stable system"},
            {
                "role": "user",
                "content": "stable contract",
                "_agent_libos_cache_stable": True,
            },
            {"role": "user", "content": "dynamic"},
        ]

        payload = client._chat_payload(messages, 0.0, 100)
        client._finalize_prompt_cache_request(payload)
        encoded = json.dumps(payload, sort_keys=True)

        assert payload["prompt_cache_options"] == {"mode": "implicit"}
        assert payload["prompt_cache_key"].startswith("alibos:v2:")
        assert "prompt_cache_breakpoint" not in encoded
        assert "_agent_libos_cache_stable" not in encoded

        appended_messages = [
            {"role": "system", "content": "stable system"},
            {
                "role": "user",
                "content": "stable contract\nnew appended requirement",
                "_agent_libos_cache_stable": True,
            },
            {"role": "user", "content": "different dynamic tail"},
        ]
        appended_payload = client._chat_payload(appended_messages, 0.0, 100)
        client._finalize_prompt_cache_request(appended_payload)
        assert appended_payload["prompt_cache_key"] == payload["prompt_cache_key"]

        changed_system = [
            {"role": "system", "content": "changed image instructions"},
            *appended_messages[1:],
        ]
        changed_payload = client._chat_payload(changed_system, 0.0, 100)
        client._finalize_prompt_cache_request(changed_payload)
        assert changed_payload["prompt_cache_key"] != payload["prompt_cache_key"]

    @pytest.mark.parametrize("api", ["chat", "responses"])
    def test_v2_explicit_cache_breakpoint_can_split_stable_and_dynamic_text(
        self,
        api: str,
    ) -> None:
        stable = "append-only materialized task context"
        dynamic = "\n\nCurrent runtime state: changed"
        client = LLMClient(
            model="gpt-5.6",
            api_key="key",
            api_mode=api,
            prompt_cache_key="tenant-domain",
            prompt_cache_mode="explicit",
        )
        messages = [
            {"role": "system", "content": "stable system"},
            {
                "role": "user",
                "content": stable + dynamic,
                "_agent_libos_cache_stable_prefix_chars": len(stable),
            },
        ]

        payload = (
            client._chat_payload(messages, 0.0, 100)
            if api == "chat"
            else client._responses_payload(messages, 0.0, 100)
        )
        client._finalize_prompt_cache_request(payload)
        encoded = json.dumps(payload, sort_keys=True)

        assert encoded.count("prompt_cache_breakpoint") == 1
        assert "_agent_libos_cache_stable_prefix_chars" not in encoded
        container = payload["messages"] if api == "chat" else payload["input"]
        user = next(item for item in container if item.get("role") == "user")
        assert user["content"][0]["text"] == stable
        assert user["content"][0]["prompt_cache_breakpoint"] == {
            "mode": "explicit"
        }
        assert user["content"][1]["text"] == dynamic

    def test_v2_cache_policy_requires_key_and_rejects_legacy_retention(self) -> None:
        with pytest.raises(LLMError, match="prompt_cache_key is required"):
            LLMClient(
                model="gpt-5.6",
                api_key="key",
                prompt_cache_mode="explicit",
            )
        with pytest.raises(LLMError, match="cannot be combined"):
            LLMClient(
                model="gpt-5.6",
                api_key="key",
                prompt_cache_key="domain",
                prompt_cache_mode="implicit",
                prompt_cache_retention="24h",
            )

    def test_v2_cache_rejection_removes_the_whole_option_group(self) -> None:
        client = LLMClient(
            model="gpt-5.6",
            api_key="key",
            prompt_cache_key="domain",
            prompt_cache_mode="explicit",
        )
        request = {
            "model": "gpt-5.6",
            "prompt_cache_key": "alibos:v2:digest",
            "prompt_cache_options": {"mode": "explicit"},
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "stable",
                            "prompt_cache_breakpoint": {"mode": "explicit"},
                        }
                    ],
                }
            ],
        }

        retry = client._compatibility_retry_payload(
            request,
            Exception("unknown parameter prompt_cache_options"),
            api="chat",
        )

        assert retry == {
            "model": "gpt-5.6",
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "stable"}],
                }
            ],
        }

    def test_compatibility_retry_reports_only_cache_options_actually_sent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        chat_completion = SimpleNamespace(
            id='chatcmpl_compat_options',
            model='compat-model',
            choices=[
                SimpleNamespace(
                    finish_reason='stop',
                    message=SimpleNamespace(content='ok', tool_calls=[]),
                )
            ],
        )
        completions = FakeChatCompletions(
            [
                Exception('unknown parameter prompt_cache_key'),
                Exception('unknown parameter prompt_cache_retention'),
                Exception('unknown parameter safety_identifier'),
                chat_completion,
            ]
        )
        fake = FakeAsyncOpenAI(chat=FakeChat(completions))
        client = LLMClient(
            base_url='https://example.com/compatible/v1',
            model='compat-model',
            api_key='key',
            api_mode='chat',
            allow_custom_base_url=True,
            safety_identifier='session-safe',
            prompt_cache_key='cache-key',
            prompt_cache_retention='24h',
        )
        client._async_client = fake
        monkeypatch.setattr(llm_client_module, '_is_openai_sdk_error', lambda _exc: True)

        completion = asyncio.run(
            client.acomplete_with_metadata(
                [{'role': 'user', 'content': 'answer'}],
                json_mode=False,
            )
        )

        assert len(completions.payloads) == 4
        assert 'prompt_cache_key' in completions.payloads[0]
        assert 'prompt_cache_key' not in completions.payloads[1]
        assert 'prompt_cache_retention' not in completions.payloads[2]
        assert 'safety_identifier' not in completions.payloads[3]
        assert completion.provider_request_options == {
            'prompt_cache_key_sent': False,
            'prompt_cache_retention': None,
            'prompt_cache_options_sent': False,
            'safety_identifier_sent': False,
        }
        assert completion.compatibility_removed_options == [
            'prompt_cache_key',
            'prompt_cache_retention',
            'safety_identifier',
        ]

    def test_json_instruction_is_independent_of_dynamic_json_text(self) -> None:
        client = LLMClient(
            model='gpt-test',
            api_key='key',
            api_mode='chat',
            defaults=LLMDefaults(json_instruction='Return canonical JSON.'),
        )
        without_keyword = client._messages_with_json_instruction(
            [
                {'role': 'system', 'content': 'Stable policy.'},
                {'role': 'user', 'content': 'Answer the request.'},
            ]
        )
        with_keyword = client._messages_with_json_instruction(
            [
                {'role': 'system', 'content': 'Stable policy.'},
                {'role': 'user', 'content': 'The dynamic tool returned JSON.'},
            ]
        )
        tool_keyword = client._messages_with_json_instruction(
            [
                {'role': 'system', 'content': 'Stable policy.'},
                {'role': 'tool', 'content': 'Return canonical JSON.'},
            ]
        )

        expected_system = 'Stable policy. Return canonical JSON.'
        assert without_keyword[0]['content'] == expected_system
        assert with_keyword[0]['content'] == expected_system
        assert tool_keyword[0]['content'] == expected_system

        already_static = client._messages_with_json_instruction(
            [
                {'role': 'developer', 'content': expected_system},
                {'role': 'user', 'content': 'anything'},
            ]
        )
        assert already_static == [
            {'role': 'developer', 'content': expected_system},
            {'role': 'user', 'content': 'anything'},
        ]

    def test_responses_bounds_native_and_plain_tool_outputs_with_metadata(self) -> None:
        response = SimpleNamespace(
            id='resp_bounded_tool_output',
            model='gpt-test',
            output_text='',
            output=[],
        )
        defaults = LLMDefaults(tool_output_prompt_max_chars=128)
        large_output = 'x' * 1_000

        native_fake = FakeAsyncOpenAI(responses=FakeResponses(response))
        native = LLMClient(
            model='gpt-test',
            api_key='key',
            api_mode='responses',
            store=True,
            defaults=defaults,
        )
        native._async_client = native_fake
        asyncio.run(
            native.acomplete_action(
                messages=[
                    {'role': 'tool', 'tool_call_id': 'call_1', 'content': large_output},
                    {'role': 'user', 'content': 'continue'},
                ],
                tools=[],
                previous_response_id='resp_previous',
            )
        )
        native_output = native_fake.responses.payloads[0]['input'][0]['output']
        assert len(native_output) <= 128
        assert 'tool_output_omitted' in native_output
        assert 'original_chars=1000' in native_output
        assert 'omitted_chars=' in native_output

        plain_fake = FakeAsyncOpenAI(responses=FakeResponses(response))
        plain = LLMClient(
            model='gpt-test',
            api_key='key',
            api_mode='responses',
            store=False,
            defaults=defaults,
        )
        plain._async_client = plain_fake
        asyncio.run(
            plain.acomplete_action(
                messages=[
                    {'role': 'tool', 'tool_call_id': 'call_1', 'content': large_output},
                    {'role': 'user', 'content': 'continue'},
                ],
                tools=[],
                previous_response_id='resp_previous',
            )
        )
        plain_output = plain_fake.responses.payloads[0]['input'][0]['content']
        assert len(plain_output) <= 128
        assert plain_output.startswith('Tool output (call_id=call_1):\n')
        assert 'tool_output_omitted' in plain_output
        assert 'original_chars=1000' in plain_output

    def test_openai_responses_payload_preserves_tool_outputs_or_breaks_state_chain(self) -> None:
        response = SimpleNamespace(id='resp_tool_output', model='gpt-test', output_text='', output=[])
        fake = FakeAsyncOpenAI(responses=FakeResponses(response))
        client = LLMClient(model='gpt-test', api_key='key', api_mode='responses', store=True)
        client._async_client = fake

        asyncio.run(
            client.acomplete_action(
                messages=[
                    {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call_1'}]},
                    {'role': 'tool', 'tool_call_id': 'call_1', 'content': '{"ok": true}'},
                    {'role': 'user', 'content': 'continue'},
                ],
                tools=[{'type': 'function', 'function': {'name': 'process_exit', 'description': 'Exit.', 'parameters': {'type': 'object', 'properties': {}}}}],
                previous_response_id='resp_prev',
            )
        )

        payload = fake.responses.payloads[0]
        assert payload['previous_response_id'] == 'resp_prev'
        assert {'type': 'function_call_output', 'call_id': 'call_1', 'output': '{"ok": true}'} in payload['input']

        no_store_fake = FakeAsyncOpenAI(responses=FakeResponses(response))
        no_store_client = LLMClient(model='gpt-test', api_key='key', api_mode='responses', store=False)
        no_store_client._async_client = no_store_fake
        asyncio.run(
            no_store_client.acomplete_action(
                messages=[
                    {'role': 'tool', 'tool_call_id': 'call_1', 'content': '{"ok": true}'},
                    {'role': 'user', 'content': 'continue'},
                ],
                tools=[{'type': 'function', 'function': {'name': 'process_exit', 'description': 'Exit.', 'parameters': {'type': 'object', 'properties': {}}}}],
                previous_response_id='resp_prev',
            )
        )
        no_store_payload = no_store_fake.responses.payloads[0]
        assert 'previous_response_id' not in no_store_payload
        assert not any(item.get('type') == 'function_call_output' for item in no_store_payload['input'])
        assert no_store_payload['input'][0] == {
            'role': 'user',
            'content': 'Tool output (call_id=call_1):\n{"ok": true}',
        }

        custom_fake = FakeAsyncOpenAI(responses=FakeResponses(response))
        custom_client = LLMClient(
            base_url='https://example.com/compatible/v1',
            model='compat-model',
            api_key='key',
            api_mode='responses',
            allow_custom_base_url=True,
            store=True,
        )
        custom_client._async_client = custom_fake
        asyncio.run(
            custom_client.acomplete_action(
                messages=[
                    {'role': 'tool', 'tool_call_id': 'call_1', 'content': '{"ok": true}'},
                    {'role': 'user', 'content': 'continue'},
                ],
                tools=[{'type': 'function', 'function': {'name': 'process_exit', 'description': 'Exit.', 'parameters': {'type': 'object', 'properties': {}}}}],
                previous_response_id='resp_prev',
            )
        )
        custom_payload = custom_fake.responses.payloads[0]
        assert 'previous_response_id' not in custom_payload
        assert not any(item.get('type') == 'function_call_output' for item in custom_payload['input'])

        missing_call_fake = FakeAsyncOpenAI(responses=FakeResponses(response))
        missing_call_client = LLMClient(model='gpt-test', api_key='key', api_mode='responses', store=True)
        missing_call_client._async_client = missing_call_fake
        asyncio.run(
            missing_call_client.acomplete_action(
                messages=[{'role': 'tool', 'content': '{"ok": true}'}],
                tools=[{'type': 'function', 'function': {'name': 'process_exit', 'description': 'Exit.', 'parameters': {'type': 'object', 'properties': {}}}}],
                previous_response_id='resp_prev',
            )
        )
        missing_call_payload = missing_call_fake.responses.payloads[0]
        assert 'previous_response_id' not in missing_call_payload
        assert not any(item.get('type') == 'function_call_output' for item in missing_call_payload['input'])
        assert missing_call_payload['input'][0] == {'role': 'user', 'content': 'Tool output:\n{"ok": true}'}

    def test_auto_mode_uses_chat_for_custom_base_url(self) -> None:
        chat_completion = SimpleNamespace(id='chatcmpl_123', model='compat-model', usage=SimpleNamespace(prompt_tokens=7, completion_tokens=2, total_tokens=9), choices=[SimpleNamespace(finish_reason='tool_calls', message=SimpleNamespace(content='', reasoning_content='select process_exit', tool_calls=[SimpleNamespace(id='tool_1', function=SimpleNamespace(name='process_exit', arguments='{"payload":{"ok":true}}'))]))])
        fake = FakeAsyncOpenAI(chat=FakeChat(FakeChatCompletions(chat_completion)))
        client = LLMClient(base_url='https://example.com/compatible/v1', model='compat-model', api_key='key', api_mode='auto', allow_custom_base_url=True)
        client._async_client = fake
        completion = asyncio.run(client.acomplete_action(messages=[{'role': 'user', 'content': 'exit'}], tools=[{'type': 'function', 'function': {'name': 'process_exit', 'description': 'Exit.', 'parameters': {'type': 'object', 'properties': {}}}}]))
        assert fake.chat.completions.payloads[0]['model'] == 'compat-model'
        assert fake.chat.completions.payloads[0]['parallel_tool_calls'] is False
        assert fake.chat.completions.payloads[0]['tools'][0]['function']['strict'] is True
        assert not fake.responses.payloads
        assert completion.api == 'chat'
        assert completion.tool_calls[0]['name'] == 'process_exit'
        assert completion.usage['total_tokens'] == 9
        assert completion.reasoning == {
            'reasoning_content': 'select process_exit',
        }
        assert completion.provider_trace is not None
        assert completion.provider_trace['attempts'][0]['reasoning']['blocks'] == [
            {
                'type': 'reasoning_text',
                'text': 'select process_exit',
                'source': 'reasoning_content',
            }
        ]

    def test_chat_trace_collects_all_allowlisted_reasoning_fields_only(self) -> None:
        message = SimpleNamespace(
            content="ok",
            reasoning="reasoning text",
            thinking_content="thinking text",
            chain_of_thought="must not be retained",
            additional_kwargs={
                "reasoning": "duplicate must not replace direct field",
                "reasoning_content": "summary text",
                "hidden_cot": "must not be retained",
            },
            tool_calls=[],
        )
        completion = SimpleNamespace(
            id="chat_reasoning_fields",
            model="gpt-test",
            choices=[SimpleNamespace(finish_reason="stop", message=message)],
        )
        fake = FakeAsyncOpenAI(chat=FakeChat(FakeChatCompletions(completion)))
        client = LLMClient(model="gpt-test", api_key="key", api_mode="chat")
        client._async_client = fake

        result = asyncio.run(
            client.acomplete_with_metadata(
                messages=[{"role": "user", "content": "answer"}],
                json_mode=False,
            )
        )

        assert result.reasoning == {
            "reasoning": "reasoning text",
            "reasoning_content": "summary text",
            "thinking_content": "thinking text",
        }
        assert result.provider_trace is not None
        blocks = result.provider_trace["attempts"][0]["reasoning"]["blocks"]
        assert [(block["source"], block["text"]) for block in blocks] == [
            ("reasoning", "reasoning text"),
            ("reasoning_content", "summary text"),
            ("thinking_content", "thinking text"),
        ]
        serialized = json.dumps(result.provider_trace, sort_keys=True)
        assert "must not be retained" not in serialized
        assert "duplicate must not replace" not in serialized

    def test_chat_action_rejects_truncated_tool_call(self) -> None:
        truncated = SimpleNamespace(
            id="chatcmpl_truncated",
            model="compat-model",
            choices=[
                SimpleNamespace(
                    finish_reason="length",
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="tool_truncated",
                                function=SimpleNamespace(
                                    name="process_exit",
                                    arguments="{}",
                                ),
                            )
                        ],
                    ),
                )
            ],
        )
        fake = FakeAsyncOpenAI(chat=FakeChat(FakeChatCompletions(truncated)))
        client = LLMClient(
            base_url="https://example.com/compatible/v1",
            model="compat-model",
            api_key="key",
            api_mode="chat",
            allow_custom_base_url=True,
            enable_thinking=False,
        )
        client._async_client = fake

        with pytest.raises(LLMError) as raised:
            asyncio.run(
                client.acomplete_action(
                    messages=[{"role": "user", "content": "exit"}],
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "process_exit",
                                "description": "Exit.",
                                "parameters": {"type": "object", "properties": {}},
                            },
                        }
                    ],
                )
            )
        assert "length" not in str(raised.value)
        assert str(raised.value).startswith("llm_error: LLMError (correlation_id=")

    def test_custom_chat_empty_response_retries_with_thinking_disabled(self) -> None:
        empty = SimpleNamespace(id='chatcmpl_empty', model='compat-model', choices=[SimpleNamespace(finish_reason='length', message=SimpleNamespace(content='', tool_calls=[]))])
        ok = SimpleNamespace(id='chatcmpl_ok', model='compat-model', choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content='OK', tool_calls=[]))])
        completions = FakeChatCompletions([empty, ok])
        fake = FakeAsyncOpenAI(chat=FakeChat(completions))
        client = LLMClient(base_url='https://example.com/compatible/v1', model='compat-model', api_key='key', api_mode='chat', allow_custom_base_url=True)
        client._async_client = fake
        content = asyncio.run(client.acomplete([{'role': 'user', 'content': 'say OK'}], json_mode=False))
        assert content == 'OK'
        assert len(completions.payloads) == 2
        assert completions.payloads[1]['extra_body'] == {'enable_thinking': False}

    def test_custom_base_url_requires_explicit_opt_in(self) -> None:
        with pytest.raises(LLMError, match='custom endpoint'):
            LLMClient(base_url='https://example.com/compatible/v1', model='compat-model', api_key='key')
        with pytest.raises(LLMError, match='custom endpoint'):
            LLMClient(base_url='https://api.openai.com.evil.example/v1', model='compat-model', api_key='key')

        client = LLMClient(
            base_url='https://example.com/compatible/v1',
            model='compat-model',
            api_key='key',
            allow_custom_base_url=True,
        )
        assert client.base_url == 'https://example.com/compatible/v1'

    def test_ambient_base_url_is_validated_and_frozen(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_BASE_URL", "https://attacker.invalid/v1")
        with pytest.raises(LLMError, match="custom endpoint"):
            LLMClient(model="gpt-test", api_key="key")

        client = LLMClient(
            model="gpt-test",
            api_key="key",
            allow_custom_base_url=True,
        )
        assert client.base_url == "https://attacker.invalid/v1"
        assert client._use_responses_api() is False

        monkeypatch.setenv("OPENAI_BASE_URL", "https://changed.invalid/v1")
        assert client._client_kwargs()["base_url"] == "https://attacker.invalid/v1"

    def test_absent_ambient_base_url_freezes_default_openai_endpoint(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        client = LLMClient(model="gpt-test", api_key="key")

        monkeypatch.setenv("OPENAI_BASE_URL", "https://attacker.invalid/v1")

        assert client._client_kwargs()["base_url"] == "https://api.openai.com/v1"
        assert client._use_responses_api() is True

    def test_api_key_env_can_be_scoped_per_client(self, monkeypatch) -> None:
        monkeypatch.setenv('OPENAI_API_KEY', 'global-key')
        monkeypatch.delenv('PROFILE_API_KEY', raising=False)
        client = LLMClient(model='profile-model', api_key_env='PROFILE_API_KEY')
        with pytest.raises(LLMError, match='PROFILE_API_KEY'):
            client._client_kwargs()

        monkeypatch.setenv('PROFILE_API_KEY', 'profile-key')
        assert client._client_kwargs()['api_key'] == 'profile-key'

    def test_from_env_reads_openai_request_option_environment(self, monkeypatch) -> None:
        monkeypatch.setenv('OPENAI_API_KEY', 'host-key')
        monkeypatch.setenv('OPENAI_MODEL', 'host-model')
        monkeypatch.setenv('OPENAI_SAFETY_IDENTIFIER', 'safe-session')
        monkeypatch.setenv('OPENAI_PROMPT_CACHE_KEY', 'cache-key')
        monkeypatch.setenv('OPENAI_PROMPT_CACHE_RETENTION', 'in-memory')
        monkeypatch.setenv('OPENAI_RESPONSES_PREVIOUS_RESPONSE_ID', 'true')
        monkeypatch.setenv('OPENAI_PARALLEL_TOOL_CALLS', 'true')
        monkeypatch.setenv('OPENAI_FALLBACK_JSON_ACTIONS', 'true')
        monkeypatch.setenv('OPENAI_ENABLE_THINKING', 'true')
        monkeypatch.setenv('OPENAI_ORG_ID', 'org-one')
        monkeypatch.setenv('OPENAI_PROJECT_ID', 'project-one')

        client = LLMClient.from_env()

        assert client.safety_identifier == 'safe-session'
        assert client.prompt_cache_key == 'cache-key'
        assert client.prompt_cache_retention == 'in_memory'
        assert client.responses_previous_response_id is True
        assert client.parallel_tool_calls is True
        assert client.fallback_json_actions is True
        assert client._extra_body() == {'enable_thinking': True}
        assert client.organization == 'org-one'
        assert client.project == 'project-one'
        assert client.inherit_ambient_openai_sdk_config is False

        monkeypatch.setenv('OPENAI_ENABLE_THINKING', 'false')
        monkeypatch.setenv('OPENAI_ORG_ID', 'org-two')
        monkeypatch.setenv('OPENAI_PROJECT_ID', 'project-two')

        assert client._extra_body() == {'enable_thinking': True}
        assert client._client_kwargs()['organization'] == 'org-one'
        assert client._client_kwargs()['project'] == 'project-one'

    def test_from_env_reads_v2_prompt_cache_mode_and_ttl(self, monkeypatch) -> None:
        monkeypatch.setenv('OPENAI_API_KEY', 'host-key')
        monkeypatch.setenv('OPENAI_MODEL', 'gpt-5.6')
        monkeypatch.setenv('OPENAI_PROMPT_CACHE_KEY', 'tenant-domain')
        monkeypatch.setenv('OPENAI_PROMPT_CACHE_MODE', 'explicit')
        monkeypatch.setenv('OPENAI_PROMPT_CACHE_TTL', '30m')
        monkeypatch.delenv('OPENAI_PROMPT_CACHE_RETENTION', raising=False)

        client = LLMClient.from_env()

        assert client.prompt_cache_mode == 'explicit'
        assert client.prompt_cache_ttl == '30m'

    def test_from_env_does_not_implicitly_load_workspace_dotenv(self, tmp_path, monkeypatch) -> None:
        (tmp_path / '.env').write_text(
            'OPENAI_BASE_URL=https://example.com/steal/v1\nOPENAI_MODEL=from-dotenv\n',
            encoding='utf-8',
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('OPENAI_API_KEY', 'host-key')
        monkeypatch.setenv('OPENAI_MODEL', 'host-model')
        monkeypatch.delenv('OPENAI_BASE_URL', raising=False)

        client = LLMClient.from_env()

        assert client.model == 'host-model'
        assert client.base_url is None

    def test_isolated_profile_strips_openai_sdk_ambient_custom_headers(self, monkeypatch) -> None:
        monkeypatch.setenv(
            'OPENAI_CUSTOM_HEADERS',
            'X-Ambient-Secret: do-not-send\nOpenAI-Organization: injected-org',
        )
        monkeypatch.setenv('OPENAI_ADMIN_KEY', 'ambient-admin-key')
        monkeypatch.setenv('OPENAI_WEBHOOK_SECRET', 'ambient-webhook-secret')
        client = LLMClient(
            model='isolated-model',
            api_key='profile-key',
            inherit_ambient_openai_sdk_config=False,
        )

        sync_sdk = client._client_or_raise()
        async_sdk = client._async_client_or_raise()
        try:
            assert 'X-Ambient-Secret' not in sync_sdk.default_headers
            assert 'X-Ambient-Secret' not in async_sdk.default_headers
            assert sync_sdk.default_headers['OpenAI-Organization'] == ''
            assert async_sdk.default_headers['OpenAI-Organization'] == ''
            assert sync_sdk.admin_api_key is None
            assert async_sdk.admin_api_key is None
            assert sync_sdk.webhook_secret is None
            assert async_sdk.webhook_secret is None
            assert sync_sdk.max_retries == 0
            assert async_sdk.max_retries == 0
            assert sync_sdk._client.follow_redirects is False
            assert async_sdk._client.follow_redirects is False
        finally:
            client.close()
            asyncio.run(async_sdk.close())

    def test_default_profile_does_not_inherit_unsupported_openai_sdk_state(self, monkeypatch) -> None:
        monkeypatch.setenv('OPENAI_CUSTOM_HEADERS', 'X-Ambient-Secret: do-not-send')
        monkeypatch.setenv('OPENAI_ADMIN_KEY', 'ambient-admin-key')
        monkeypatch.setenv('OPENAI_WEBHOOK_SECRET', 'ambient-webhook-secret')
        client = LLMClient(model='default-model', api_key='profile-key')

        sdk = client._client_or_raise()
        try:
            assert 'X-Ambient-Secret' not in sdk.default_headers
            assert sdk.admin_api_key is None
            assert sdk.webhook_secret is None
            assert sdk.max_retries == 0
            assert sdk._client.follow_redirects is False
        finally:
            client.close()

    def test_sdk_header_isolation_does_not_depend_on_a_second_environment_read(self, monkeypatch) -> None:
        monkeypatch.delenv('OPENAI_CUSTOM_HEADERS', raising=False)
        sdk = SimpleNamespace(
            _custom_headers={'X-Captured-Before-Environment-Change': 'do-not-send'},
            admin_api_key='ambient-admin-key',
            webhook_secret='ambient-webhook-secret',
            max_retries=0,
            _client=SimpleNamespace(follow_redirects=True),
        )

        normalized = LLMClient._normalize_openai_sdk_client(sdk)

        assert normalized is sdk
        assert sdk._custom_headers == {}
        assert sdk.admin_api_key is None
        assert sdk.webhook_secret is None
        assert sdk._client.follow_redirects is False

    @pytest.mark.parametrize(
        "sdk",
        [
            SimpleNamespace(_custom_headers={}),
            SimpleNamespace(_custom_headers={}, max_retries=0),
        ],
    )
    def test_sdk_retry_and_redirect_controls_fail_closed(self, sdk: Any) -> None:
        with pytest.raises(LLMError):
            LLMClient._normalize_openai_sdk_client(sdk)

    def test_from_env_and_requests_use_configured_llm_defaults(self, monkeypatch) -> None:
        config = AgentLibOSConfig(
            llm=LLMDefaults(
                temperature=0.7,
                max_tokens=123,
                timeout_s=9.0,
                max_retries=4,
                api_mode='chat',
                store=True,
                json_instruction='Return valid JSON.',
            )
        )
        monkeypatch.setenv('OPENAI_API_KEY', 'host-key')
        monkeypatch.setenv('OPENAI_MODEL', 'host-model')
        monkeypatch.delenv('OPENAI_API_MODE', raising=False)
        chat_completion = SimpleNamespace(
            id='chatcmpl_config',
            model='host-model',
            choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content='{"ok":true}', tool_calls=[]))],
        )
        fake = FakeAsyncOpenAI(chat=FakeChat(FakeChatCompletions(chat_completion)))

        client = LLMClient.from_env(config=config)
        client._async_client = fake
        content = asyncio.run(client.acomplete([{'role': 'user', 'content': 'answer'}], json_mode=True))

        payload = fake.chat.completions.payloads[0]
        assert client.timeout == 9.0
        assert client.max_retries == 4
        assert client.api_mode == 'chat'
        assert client.store is True
        assert content == '{"ok":true}'
        assert payload['temperature'] == 0.7
        assert payload['max_completion_tokens'] == 123
        assert payload['store'] is True
        assert payload['messages'][0]['content'] == 'Return valid JSON.'

    def test_from_env_explicit_custom_base_url_requires_allowance(self, tmp_path, monkeypatch) -> None:
        env_file = tmp_path / 'llm.env'
        env_file.write_text(
            'OPENAI_BASE_URL=https://example.com/compatible/v1\nOPENAI_MODEL=compat-model\nOPENAI_API_KEY=key\n',
            encoding='utf-8',
        )
        monkeypatch.delenv('AGENT_LIBOS_ALLOW_CUSTOM_LLM_BASE_URL', raising=False)

        with pytest.raises(LLMError, match='custom endpoint'):
            LLMClient.from_env(env_file)

        client = LLMClient.from_env(env_file, allow_custom_base_url=True)
        assert client.base_url == 'https://example.com/compatible/v1'

    def test_responses_max_output_token_incompatibility_fails_closed(self) -> None:
        client = LLMClient(model='gpt-test', api_key='key', api_mode='responses')
        retry = client._compatibility_retry_payload(
            {'model': 'gpt-test', 'max_output_tokens': 100},
            Exception('unknown parameter max_output_tokens'),
            api='responses',
        )
        assert retry is None

    def test_new_prompt_cache_options_incompatibility_can_downgrade(self) -> None:
        client = LLMClient(model='gpt-test', api_key='key', api_mode='responses')
        retry = client._compatibility_retry_payload(
            {
                'model': 'gpt-test',
                'prompt_cache_options': {'mode': 'implicit', 'ttl': '30m'},
            },
            Exception('unknown parameter prompt_cache_options'),
            api='responses',
        )

        assert retry == {'model': 'gpt-test'}

    def test_strict_tool_incompatibility_retry_removes_strict_fields(self) -> None:
        client = LLMClient(model='gpt-test', api_key='key', api_mode='chat')
        retry = client._compatibility_retry_payload(
            {
                'model': 'gpt-test',
                'tools': [
                    {
                        'type': 'function',
                        'function': {
                            'name': 'tool',
                            'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False},
                            'strict': True,
                        },
                    },
                    {'type': 'function', 'name': 'responses_style', 'strict': True},
                ],
            },
            Exception('unknown parameter strict'),
            api='chat',
        )

        assert retry is not None
        assert 'strict' not in retry['tools'][0]['function']
        assert 'strict' not in retry['tools'][1]

    def test_tool_protocol_rejection_fails_when_json_fallback_is_disabled(self) -> None:
        fake = FakeAsyncOpenAI(
            chat=FakeChat(
                FakeChatCompletions(LLMError("provider rejected tools parameter"))
            )
        )
        client = LLMClient(
            model="gpt-test",
            api_key="key",
            api_mode="chat",
        )
        client._async_client = fake

        with pytest.raises(LLMError, match="rejected tools"):
            asyncio.run(
                client.acomplete_action(
                    messages=[{"role": "user", "content": "exit"}],
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "process_exit",
                                "description": "Exit.",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                )
            )

        assert len(fake.chat.completions.payloads) == 1

    def test_tool_protocol_rejection_opt_in_preserves_fallback_metadata(self) -> None:
        fallback_completion = SimpleNamespace(
            id="chatcmpl_fallback",
            _request_id="req_fallback",
            model="fallback-model",
            usage=SimpleNamespace(
                prompt_tokens=21,
                completion_tokens=4,
                total_tokens=25,
                prompt_tokens_details=SimpleNamespace(cached_tokens=13),
            ),
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content='{"action":"process_exit","payload":{"done":true}}',
                        tool_calls=[],
                    ),
                )
            ],
        )
        fake = FakeAsyncOpenAI(
            chat=FakeChat(
                FakeChatCompletions(
                    [
                        LLMError("provider rejected tool_choice and tools"),
                        fallback_completion,
                    ]
                )
            )
        )
        client = LLMClient(
            model="gpt-test",
            api_key="key",
            api_mode="chat",
            fallback_json_actions=True,
        )
        client._async_client = fake

        completion = asyncio.run(
            client.acomplete_action(
                messages=[{"role": "user", "content": "exit"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "process_exit",
                            "description": "Exit.",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            )
        )

        assert completion.fallback_json_action_used is True
        assert completion.model == "fallback-model"
        assert completion.request_id == "req_fallback"
        assert completion.response_id == "chatcmpl_fallback"
        assert completion.usage["total_tokens"] == 25
        assert completion.raw is fallback_completion
        assert "tools" in fake.chat.completions.payloads[0]
        assert "tools" not in fake.chat.completions.payloads[1]
        assert "tool_choice" not in fake.chat.completions.payloads[1]
        assert completion.provider_trace is not None
        assert completion.provider_trace["selected_attempt"] == 2
        assert [
            attempt["kind"] for attempt in completion.provider_trace["attempts"]
        ] == ["initial", "json_action_fallback"]
        assert [
            attempt["status"] for attempt in completion.provider_trace["attempts"]
        ] == ["error", "ok"]

    def test_auto_mode_records_responses_to_chat_fallback_trace(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class EndpointNotFound(Exception):
            status_code = 404
            response = SimpleNamespace(headers={})

        class FailingResponses:
            def __init__(self) -> None:
                self.payloads: list[dict[str, Any]] = []

            async def create(self, **payload: Any) -> Any:
                self.payloads.append(payload)
                raise EndpointNotFound("Responses endpoint not found")

        chat_completion = SimpleNamespace(
            id="chat_fallback",
            model="gpt-test",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="ok", tool_calls=[]),
                )
            ],
        )
        responses = FailingResponses()
        chat = FakeChat(FakeChatCompletions(chat_completion))
        fake = FakeAsyncOpenAI(responses=responses, chat=chat)
        client = LLMClient(model="gpt-test", api_key="key", api_mode="auto")
        client._async_client = fake
        monkeypatch.setattr(llm_client_module, "_is_openai_sdk_error", lambda _exc: True)

        completion = asyncio.run(
            client.acomplete_with_metadata(
                messages=[{"role": "user", "content": "answer"}],
                json_mode=False,
            )
        )

        assert len(responses.payloads) == 1
        assert len(chat.completions.payloads) == 1
        assert completion.provider_trace is not None
        assert completion.provider_trace["selected_attempt"] == 2
        assert [
            (attempt["api"], attempt["kind"], attempt["status"])
            for attempt in completion.provider_trace["attempts"]
        ] == [
            ("responses", "initial", "error"),
            ("chat", "responses_to_chat", "ok"),
        ]

    def test_close_releases_cached_sync_and_async_clients(self) -> None:
        client = LLMClient(model='gpt-test', api_key='key')
        sync = ClosableClient()
        async_client = AsyncClosableClient()
        client._client = sync
        client._async_client = async_client

        client.close()

        assert sync.closed
        assert async_client.closed
        assert client._client is None
        assert client._async_client is None

    def test_close_can_release_async_client_inside_running_event_loop(self) -> None:
        async def run() -> bool:
            client = LLMClient(model='gpt-test', api_key='key')
            async_client = AsyncClosableClient()
            client._async_client = async_client

            client.close()

            assert client._async_client is None
            return async_client.closed

        assert asyncio.run(run()) is True

    def test_aclose_releases_cached_sync_and_async_clients(self) -> None:
        async def run() -> tuple[bool, bool]:
            client = LLMClient(model='gpt-test', api_key='key')
            sync = ClosableClient()
            async_client = AsyncClosableClient()
            client._client = sync
            client._async_client = async_client

            await client.aclose()

            assert client._client is None
            assert client._async_client is None
            return sync.closed, async_client.closed

        assert asyncio.run(run()) == (True, True)

    def test_aclose_offloads_cached_sync_client_timer_barrier(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        close_threads: list[int] = []

        class BlockingSyncClient:
            def close(self) -> None:
                close_threads.append(threading.get_ident())
                entered.set()
                assert release.wait(timeout=1)

        async def run() -> None:
            client = LLMClient(model="gpt-test", api_key="key")
            client._client = BlockingSyncClient()
            caller_thread = threading.get_ident()
            asyncio.get_running_loop().call_later(0.01, release.set)

            await client.aclose()

            assert entered.is_set()
            assert close_threads and close_threads[0] != caller_thread
            assert client._client is None

        asyncio.run(run())

class FakeAsyncOpenAI:

    def __init__(self, responses: Any | None=None, chat: Any | None=None):
        self.responses = responses or FakeResponses(SimpleNamespace(id='unused', model='unused', output_text='', output=[]))
        self.chat = chat or FakeChat(FakeChatCompletions(SimpleNamespace(choices=[])))

class FakeResponses:

    def __init__(self, response: Any):
        self.response = response
        self.payloads: list[dict[str, Any]] = []

    async def create(self, **payload: Any) -> Any:
        self.payloads.append(payload)
        return self.response

class FakeChat:

    def __init__(self, completions: Any):
        self.completions = completions

class FakeChatCompletions:

    def __init__(self, completion: Any):
        self.completions = list(completion) if isinstance(completion, list) else [completion]
        self.payloads: list[dict[str, Any]] = []

    async def create(self, **payload: Any) -> Any:
        self.payloads.append(payload)
        selected = self.completions.pop(0)
        if isinstance(selected, BaseException):
            raise selected
        return selected


class ClosableClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class AsyncClosableClient:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _KeepAliveChatHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    request_count = 0

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        type(self).request_count += 1
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length:
            self.rfile.read(content_length)
        payload = json.dumps(
            {
                "id": f"chatcmpl-{type(self).request_count}",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    def log_message(self, _format: str, *_args: object) -> None:
        return
