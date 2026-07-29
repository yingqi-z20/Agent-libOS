from __future__ import annotations
import pytest
import asyncio
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any
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
from agent_libos.utils.public_errors import public_error_envelope

class TestLLMClient:

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
            'requested_model': 'compat-model',
            'max_completion_tokens': 16_384,
            'generation_token_limit_parameter': 'max_completion_tokens',
            'timeout_s': 60.0,
            'enable_thinking': None,
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
        assert completion.reasoning == 'select process_exit'

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
        finally:
            client.close()

    def test_sdk_header_isolation_does_not_depend_on_a_second_environment_read(self, monkeypatch) -> None:
        monkeypatch.delenv('OPENAI_CUSTOM_HEADERS', raising=False)
        sdk = SimpleNamespace(
            _custom_headers={'X-Captured-Before-Environment-Change': 'do-not-send'},
            admin_api_key='ambient-admin-key',
            webhook_secret='ambient-webhook-secret',
        )

        normalized = LLMClient._normalize_openai_sdk_client(sdk)

        assert normalized is sdk
        assert sdk._custom_headers == {}
        assert sdk.admin_api_key is None
        assert sdk.webhook_secret is None

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

    def test_explicitly_enabled_thinking_incompatibility_fails_closed(
        self,
        monkeypatch,
    ) -> None:
        completed = SimpleNamespace(
            id='chatcmpl_unused',
            model='compat-model',
            choices=[
                SimpleNamespace(
                    finish_reason='stop',
                    message=SimpleNamespace(content='unused', tool_calls=[]),
                )
            ],
        )
        completions = FakeChatCompletions(
            [Exception('unknown parameter enable_thinking'), completed]
        )
        client = LLMClient(
            base_url='https://example.com/compatible/v1',
            model='compat-model',
            api_key='key',
            api_mode='chat',
            allow_custom_base_url=True,
            enable_thinking=True,
        )
        client._async_client = FakeAsyncOpenAI(chat=FakeChat(completions))
        monkeypatch.setattr(
            llm_client_module,
            '_is_openai_sdk_error',
            lambda _exc: True,
        )

        with pytest.raises(LLMError):
            asyncio.run(
                client.acomplete(
                    [{'role': 'user', 'content': 'answer'}],
                    json_mode=False,
                )
            )

        assert len(completions.payloads) == 1
        assert completions.payloads[0]['extra_body'] == {
            'enable_thinking': True
        }

    def test_chat_token_name_fallback_preserves_fixed_value_and_thinking_telemetry(
        self,
        monkeypatch,
    ) -> None:
        completed = SimpleNamespace(
            id='chatcmpl_compat_tokens',
            model='compat-model',
            choices=[
                SimpleNamespace(
                    finish_reason='stop',
                    message=SimpleNamespace(content='ok', tool_calls=[]),
                )
            ],
        )
        completions = FakeChatCompletions(
            [Exception('unknown parameter max_completion_tokens'), completed]
        )
        client = LLMClient(
            base_url='https://example.com/compatible/v1',
            model='compat-model',
            api_key='key',
            api_mode='chat',
            allow_custom_base_url=True,
            timeout=240.0,
            enable_thinking=True,
        )
        client._async_client = FakeAsyncOpenAI(chat=FakeChat(completions))
        monkeypatch.setattr(
            llm_client_module,
            '_is_openai_sdk_error',
            lambda _exc: True,
        )

        completion = asyncio.run(
            client.acomplete_with_metadata(
                [{'role': 'user', 'content': 'answer'}],
                max_tokens=65_536,
                json_mode=False,
            )
        )

        assert len(completions.payloads) == 2
        assert completions.payloads[0]['max_completion_tokens'] == 65_536
        assert completions.payloads[1]['max_tokens'] == 65_536
        assert completions.payloads[1]['extra_body'] == {
            'enable_thinking': True
        }
        assert completion.provider_request_options[
            'generation_token_limit_parameter'
        ] == 'max_tokens'
        assert completion.provider_request_options['requested_model'] == 'compat-model'
        assert completion.provider_request_options['max_completion_tokens'] == 65_536
        assert completion.provider_request_options['timeout_s'] == 240.0
        assert completion.provider_request_options['enable_thinking'] is True
        assert completion.compatibility_removed_options == [
            'max_completion_tokens'
        ]

    def test_requested_and_response_models_are_distinct_metadata(self) -> None:
        provider_response = SimpleNamespace(
            id='chatcmpl_routed',
            model='provider-routed-backend',
            choices=[
                SimpleNamespace(
                    finish_reason='stop',
                    message=SimpleNamespace(content='ok', tool_calls=[]),
                )
            ],
        )
        fake = FakeAsyncOpenAI(
            chat=FakeChat(FakeChatCompletions(provider_response))
        )
        client = LLMClient(
            model='qwen3.7-max',
            api_key='key',
            api_mode='chat',
        )
        client._async_client = fake

        completion = asyncio.run(
            client.acomplete_with_metadata(
                [{'role': 'user', 'content': 'answer'}],
                json_mode=False,
            )
        )

        assert completion.provider_request_options['requested_model'] == 'qwen3.7-max'
        assert completion.model == 'provider-routed-backend'

    def test_required_max_completion_tokens_rejects_parameter_name_fallback(
        self,
        monkeypatch,
    ) -> None:
        completions = FakeChatCompletions(
            [Exception('unknown parameter max_completion_tokens')]
        )
        client = LLMClient(
            base_url='https://example.com/formal-evaluation/v1',
            model='formal-model',
            api_key='key',
            api_mode='chat',
            allow_custom_base_url=True,
            require_max_completion_tokens=True,
        )
        client._async_client = FakeAsyncOpenAI(chat=FakeChat(completions))
        monkeypatch.setattr(
            llm_client_module,
            '_is_openai_sdk_error',
            lambda _exc: True,
        )

        with pytest.raises(LLMError):
            asyncio.run(
                client.acomplete_with_metadata(
                    [{'role': 'user', 'content': 'answer'}],
                    max_tokens=65_536,
                    json_mode=False,
                )
            )

        assert len(completions.payloads) == 1
        assert completions.payloads[0]['max_completion_tokens'] == 65_536
        assert 'max_tokens' not in completions.payloads[0]

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
