from __future__ import annotations

from dataclasses import replace

from agent_libos import AgentImage, Runtime
from agent_libos.models import CapabilityRight, ObjectMetadata, ObjectType, ViewMode
from agent_libos.utils.serde import dumps
from tests.runtime.test_responses_replay_executor import (
    CONFIG,
    SECRET,
    ReplayClient,
    completion,
)


SUMMARY = "MODEL_SELECTED_COMPACTION_SUMMARY_APPEARS_ONCE"


def test_model_selected_compaction_resumes_private_replay_after_child_settles() -> None:
    config = replace(
        CONFIG,
        llm_context=replace(CONFIG.llm_context, policy="llm_context_object"),
    )
    runtime = Runtime.open("local", config=config)
    try:
        client = ReplayClient([
            completion(1),
            completion(2, name="compact_process_context", arguments={
                "force": True,
                "target_tokens": 512,
                "max_chunks": 1,
                "preserve_recent_entries": 0,
            }),
            completion(3, name="process_exit", arguments={"payload": {
                "goal": SUMMARY,
                "constraints": [],
                "user_preferences": [],
                "completed": ["First echo finished."],
                "pending": ["Resume the parent process."],
                "key_references": {},
                "recent_decisions": [],
                "risks": [],
                "uncertainties": [],
                "next_steps": ["Perform the final echo."],
            }}),
            completion(4),
        ])
        runtime.llm.client = client
        image_id = "responses-model-compaction:v0"
        runtime.register_image(AgentImage(
            image_id=image_id,
            name="Responses compaction test",
            system_prompt="Echo, compact the context, then echo again.",
            default_tools=["echo", "process_exit", "compact_process_context", "get_current_time", "sleep"],
        ), actor="test")
        pid = runtime.process.spawn(image=image_id, goal="Resume after model-selected context compaction.")
        runtime.skills.activate_skill(pid, "agent-libos-runtime-session", actor=pid)
        runtime.capability.grant(
            pid, "process:spawn", [CapabilityRight.WRITE], issued_by="test",
        )
        runtime.capability.grant(
            pid, "image:context-compressor:v0", [CapabilityRight.READ], issued_by="test",
        )
        source = runtime.memory.create_object(
            pid, ObjectType.EVIDENCE, {"value": "Retain this untrusted source attribution."},
            metadata=ObjectMetadata(trust_level="untrusted", integrity="untrusted"),
        )
        process = runtime.process.get(pid)
        process.memory_view = runtime.memory.create_view(pid, [source], mode=ViewMode.READ_ONLY)
        runtime.store.update_process(process)

        first = runtime.run_process_once(pid)
        assert first["ok"], first
        original_generation = runtime.store.get_llm_context_generation(pid)
        waiting = runtime.run_process_once(pid)
        assert waiting["waiting_event"], waiting
        assert len(client.inputs) == 2
        pending = runtime.store.get_llm_pending_action(pid)
        assert pending["wait_type"] == "child"

        # The model's compaction call is a real pending replay group until its
        # compressor child finishes and the parent commits the tool result.
        _head, waiting_turn, waiting_payload = runtime.llm.replay.load_current(pid)
        assert waiting_turn.context_generation == original_generation
        assert len(waiting_payload["groups"]) == 2
        assert waiting_payload["groups"][-1]["validated"] is True
        assert waiting_payload["groups"][-1]["tool_outputs"] == []
        assert SECRET + "1" in dumps(waiting_payload)
        assert SECRET + "2" in dumps(waiting_payload)
        assert waiting_turn.source_labels["trust_level"] == "untrusted"
        assert waiting_turn.source_labels["integrity"] == "untrusted"
        original_source_refs = waiting_payload["flow_context"]["source_refs"]
        assert original_source_refs

        child_result = runtime.run_next_process_once()
        assert child_result["ok"], child_result
        assert child_result["action"]["action"] == "process_exit"
        assert len(client.inputs) == 3
        assert SECRET not in dumps(client.inputs[2])

        resumed = runtime.run_process_once(pid)
        assert resumed["result"]["ok"], resumed
        assert resumed["action"]["action"] == "compact_process_context"
        output = resumed["result"]["payload"]
        assert output["compacted"] is True
        assert len(output["compressor_pids"]) == 1
        child = runtime.process.get(output["compressor_pids"][0])
        assert child.image_id == "context-compressor:v0"
        assert len(client.inputs) == 3, "Resuming the tool must not call the provider again."
        assert runtime.store.get_llm_pending_action(pid)["status"] == "completed"
        assert runtime.store.get_llm_context_generation(pid) != original_generation

        final = runtime.run_process_once(pid)
        assert final["ok"], final
        assert len(client.inputs) == 4
        final_wire = dumps(client.inputs[3])
        assert final_wire.count(SUMMARY) == 1
        assert SECRET not in final_wire
        assert not any(item.get("call_id") in {"call_1", "call_2", "call_3"} for item in client.inputs[3])
        _head, final_turn, final_payload = runtime.llm.replay.load_current(pid)
        assert final_turn.context_generation == runtime.store.get_llm_context_generation(pid)
        assert final_turn.source_labels["trust_level"] == "untrusted"
        assert final_turn.source_labels["integrity"] == "untrusted"
        final_source_refs = final_payload["flow_context"]["source_refs"]
        assert all(ref in final_source_refs for ref in original_source_refs)
        assert len(runtime.store.list_llm_calls(pid=pid)) == 3
        assert SECRET not in dumps(runtime.store.list_llm_calls(pid=pid))
    finally:
        runtime.close()
