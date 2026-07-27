from __future__ import annotations
import pytest
import asyncio
import tempfile
import json
from agent_libos import Runtime
from agent_libos.llm.client import LLMCompletion
from scripts.human_llm_chat import CHAT_PROCESS_GOAL, EchoResponder, ModelResponder, run_chat
from scripts.llm_context_probe import last_tool_result

class TestHumanLLMChatScript:

    def test_chat_process_goal_is_plain_prompt_text(self) -> None:
        assert isinstance(CHAT_PROCESS_GOAL, str)
        assert 'Every turn MUST conclude with a tool call' in CHAT_PROCESS_GOAL
        assert 'ask_human' in CHAT_PROCESS_GOAL

    def test_chat_uses_human_question_and_output_tools(self) -> None:
        report = asyncio.run(run_chat(responder=EchoResponder(), max_turns=5, auto_messages=['hello', '/exit'], echo=False))
        assert report['process_status'] == 'exited'
        assert report['turns'] == 1
        assert report['history'] == [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'Echo: hello'}]
        assert 'Assistant: Echo: hello' in report['outputs']
        assert 'Assistant: goodbye.' in report['outputs']
        assert report['actions'] == [None, 'ask_human', 'human_output', None, 'ask_human', 'human_output', 'process_exit']

    def test_model_responder_persists_nested_text_llm_call(self) -> None:
        responder = ModelResponder.__new__(ModelResponder)
        responder.system_prompt = 'System prompt'
        responder.client = FakeTextLLMClient()
        responder._runtime = None
        responder._pid = None
        with tempfile.TemporaryDirectory() as temp_dir:
            db = f'{temp_dir}/runtime.sqlite'
            report = asyncio.run(run_chat(db=db, responder=responder, max_turns=5, auto_messages=['hello', '/exit'], echo=False))
            runtime = Runtime.open(db)
            try:
                calls = [call for call in runtime.store.list_llm_calls(report['pid']) if call.purpose == 'script_human_chat_reply']
                assert len(calls) == 1
                assert calls[0].response_content == 'model reply'
                assert calls[0].usage['total_tokens'] == 6
                assert calls[0].reasoning == {'summary': 'fake text response'}
                assert calls[0].messages[1]['content'] == 'hello'
                assert calls[0].observability['reasoning']['sha256']
                assert calls[0].observability['messages']['sha256']
                assert 'hello' in json.dumps(calls[0].__dict__, sort_keys=True)
            finally:
                runtime.close()

    def test_chat_persistent_database_can_be_reused(self, tmp_path) -> None:
        db = str(tmp_path / "chat.sqlite")

        first = asyncio.run(
            run_chat(
                db=db,
                responder=EchoResponder(),
                max_turns=1,
                auto_messages=["/exit"],
                echo=False,
            )
        )
        second = asyncio.run(
            run_chat(
                db=db,
                responder=EchoResponder(),
                max_turns=1,
                auto_messages=["/exit"],
                echo=False,
            )
        )

        assert first["process_status"] == "exited"
        assert second["process_status"] == "exited"
        assert first["pid"] != second["pid"]

    def test_model_responder_durable_error_excludes_provider_text(self, tmp_path) -> None:
        responder = ModelResponder.__new__(ModelResponder)
        responder.system_prompt = "System prompt"
        responder.client = FailingTextLLMClient()
        responder._runtime = None
        responder._pid = None
        db = str(tmp_path / "chat-error.sqlite")

        with pytest.raises(RuntimeError, match="chat process did not exit"):
            asyncio.run(
                run_chat(
                    db=db,
                    responder=responder,
                    max_turns=1,
                    auto_messages=["hello"],
                    echo=False,
                )
            )

        runtime = Runtime.open(db)
        try:
            calls = [
                call
                for call in runtime.store.list_llm_calls()
                if call.purpose == "script_human_chat_reply"
            ]
            assert len(calls) == 1
            serialized = json.dumps(calls[0].__dict__, sort_keys=True)
            assert "provider-secret-diagnostic" not in serialized
            assert "llm_provider_error" in (calls[0].error or "")
        finally:
            runtime.close()

    def test_source_context_fallback_uses_latest_result_in_stable_root_order(self) -> None:
        messages = [
            {
                'role': 'user',
                'content': '\n\n'.join(
                    [
                        json.dumps(
                            {
                                'record_type': 'object_memory_object',
                                'render_format': 'canonical_json_v1',
                                'object_oid': 'obj-old',
                                'type': 'tool_result',
                                'payload': {
                                    'tool_name': 'ask_human',
                                    'result': {'answer': 'hello'},
                                },
                            },
                            separators=(',', ':'),
                        ),
                        json.dumps(
                            {
                                'record_type': 'object_memory_object',
                                'render_format': 'canonical_json_v1',
                                'object_oid': 'obj-new',
                                'type': 'tool_result',
                                'payload': {
                                    'tool_name': 'ask_human',
                                    'result': {'answer': '/exit'},
                                },
                            },
                            separators=(',', ':'),
                        ),
                    ]
                ),
            }
        ]

        assert last_tool_result(messages, 'ask_human') == {'answer': '/exit'}

class FakeTextLLMClient:
    model = 'fake-text-model'

    def complete_with_metadata(self, messages, *, json_mode: bool) -> LLMCompletion:
        return LLMCompletion(content='model reply', tool_calls=[], raw={'id': 'fake_raw'}, api='chat', response_id='fake_resp', request_id='fake_req', model=self.model, usage={'prompt_tokens': 4, 'completion_tokens': 2, 'total_tokens': 6}, reasoning={'summary': 'fake text response'})


class FailingTextLLMClient:
    model = "failing-text-model"

    def complete_with_metadata(self, messages, *, json_mode: bool) -> LLMCompletion:
        raise RuntimeError("provider-secret-diagnostic")
