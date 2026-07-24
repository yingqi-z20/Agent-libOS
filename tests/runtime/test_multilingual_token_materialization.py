from __future__ import annotations

from agent_libos import Runtime
from agent_libos.llm.context_management import estimate_multilingual_tokens
from agent_libos.models import ObjectType


def test_materialization_does_not_underestimate_request_layer_for_cjk_text() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="验证多语言上下文预算",
        )
        handle = runtime.memory.create_object(
            pid=pid,
            object_type=ObjectType.OBSERVATION,
            payload={"text": "缓存前缀应当稳定，中文不能按四字符一个 token 估算。"},
        )
        context = runtime.memory.materialize_context(
            pid,
            runtime.memory.create_view(pid, [handle]),
            budget_tokens=100_000,
            charge_resources=False,
        )

        assert context.object_refs == [handle.oid]
        assert context.token_count >= estimate_multilingual_tokens(context.text)
    finally:
        runtime.close()
