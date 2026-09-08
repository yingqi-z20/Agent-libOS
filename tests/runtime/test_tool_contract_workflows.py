"""Execute serialized, wire-valid calls against the real broker and primitives."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import pytest

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.prompt import _compact_tool_result_payload
from agent_libos.models import AgentImage, CapabilityRight, ProcessStatus
from agent_libos.substrate import LocalResourceProviderSubstrate
from agent_libos.utils.openai_schema import openai_responses_tool_schema
from scripts.tool_contract_support import REQUIRED_FIELD_CONTRACTS, builtin_tools


@pytest.mark.parametrize("layout", ["legacy_v1", "cache_optimized_v2"])
@pytest.mark.parametrize("api", ["chat", "responses"])
def test_wire_valid_contract_workflow_preserves_state_and_authority(tmp_path: Path, layout: str, api: str) -> None:
    config = replace(
        DEFAULT_CONFIG,
        llm=replace(DEFAULT_CONFIG.llm, prompt_layout=layout),
        memory=replace(DEFAULT_CONFIG.memory, process_namespace_prefix="contract-process"),
    )
    runtime = Runtime.open("local", config=config, substrate=LocalResourceProviderSubstrate(tmp_path))
    tools = {tool.name: tool for tool in builtin_tools()}
    called: set[str] = set()
    try:
        runtime.register_image(AgentImage(
            image_id="contract-fixture:v0", name="contract-fixture", version="v0",
            default_tools=sorted(set(REQUIRED_FIELD_CONTRACTS) | {"read_text_file"}),
        ), actor="test.host")
        pid = runtime.process.spawn(image="contract-fixture:v0", goal="Verify tool contracts")

        def call(name: str, /, **args: Any) -> dict[str, Any]:
            tool = tools[name]
            chat = tool.to_openai_chat_tool(config=config, prompt_layout=layout)
            wire = openai_responses_tool_schema(chat) if api == "responses" else chat["function"]
            assert wire is not None
            parsed = tool.parse_args(args, config=config).model_dump(mode="json")
            # Strict calls must include every advertised property, including
            # null defaults, but must never include internal continuation state.
            serialized = json.dumps({key: parsed[key] for key in wire["parameters"]["properties"]})
            arguments = json.loads(serialized)
            Draft202012Validator(wire["parameters"]).validate(arguments)
            called.add(name)
            return runtime.llm.dispatch(pid, {"action": name, **arguments})

        current = runtime.memory.resolve_namespace(pid)
        literal = '{"entries":[]}'
        assert call("create_memory_object", name="literal", type="plan", namespace=None, payload=literal, immutable=False)["ok"]
        assert call("read_memory_object", name="literal", namespace=None)["payload"]["payload"] == literal
        rejected = call("append_memory_object", name="literal", namespace=None, entry={"step": "must not append"})
        assert not rejected["ok"]
        assert runtime.memory.get_object_by_name(pid, "literal").payload == literal
        assert call("create_memory_object", name="ledger", type="plan", namespace=None, payload={"entries": []}, immutable=False)["ok"]
        assert call("append_memory_object", name="ledger", namespace=None, entry=literal)["ok"]
        read = call("read_memory_object", name="ledger", namespace=None)
        assert read["payload"]["payload"] == {"entries": [literal]}
        assert runtime.memory.get_object_by_name(pid, "ledger").namespace == current
        assert call("create_memory_object", name="nullable", type="plan", namespace=None, payload=None)["ok"]
        null_read = call("read_memory_object", name="nullable", namespace=None)
        assert null_read["ok"] and null_read["payload"]["payload"] is None
        carrier = runtime.store.get_object(null_read["result_oid"]).payload
        replay = _compact_tool_result_payload(carrier, current_namespace=current)
        assert replay["result"]["representation"] == "json_value"
        assert replay["result"]["payload"] is None
        listed = call("list_memory_namespace", namespace=None)
        assert listed["ok"] and "ledger" in {item["name"] for item in listed["payload"]["objects"]}

        assert call("create_memory_namespace", namespace="project", parent_namespace=None)["ok"]
        assert call("create_memory_namespace", namespace="project/child", parent_namespace=None)["ok"]
        assert runtime.store.get_namespace("project").parent_namespace is None
        assert runtime.store.get_namespace("project/child").parent_namespace == "project"

        for path in ("source.txt", "copy.txt"):
            runtime.filesystem.grant_path(pid, path, [CapabilityRight.READ, CapabilityRight.WRITE], issued_by="test.host")
        first_write = call("write_text_file", path="source.txt", content="initial\n", expected_content_sha256="missing")
        assert first_write["ok"], first_write
        digest = call("read_text_file", path="source.txt")["payload"]["content_sha256"]
        assert not call("write_text_file", path="source.txt", content="lost update\n", expected_content_sha256="0" * 64)["ok"]
        assert (tmp_path / "source.txt").read_text() == "initial\n"
        assert call("write_text_file", path="source.txt", content="verified\n", expected_content_sha256=digest)["ok"]
        assert call("create_object_from_file", name="text", path="source.txt", namespace=None)["ok"]
        assert runtime.memory.get_object_by_name(pid, "text").namespace == current
        assert call("write_object_to_file", name="text", path="copy.txt", namespace=None)["ok"]
        assert (tmp_path / "copy.txt").read_text() == "verified\n"

        checkpoint = call("create_checkpoint", reason="Verified contract workflow", pid=None)
        assert checkpoint["ok"]
        assert call("list_checkpoints", pid=None)["ok"]
        snapshots = runtime.checkpoint.list(pid, actor=pid)
        assert len(snapshots) == 1 and snapshots[0]["pid"] == pid
        checkpoint_id = snapshots[0]["checkpoint_id"]
        assert not call("fork_checkpoint", checkpoint_id=checkpoint_id, parent_pid=None)["ok"]
        runtime.capability.grant(pid, f"checkpoint:{checkpoint_id}", [CapabilityRight.EXECUTE], issued_by="test.host")
        fork = call("fork_checkpoint", checkpoint_id=checkpoint_id, parent_pid=None)
        assert fork["ok"], fork
        fork_pid = fork["payload"]["fork_root_pid"]
        assert fork_pid != pid and runtime.process.get(fork_pid).parent_pid is None
        assert any(
            record.action == "tool.call" and record.decision.get("tool") == "fork_checkpoint"
            and record.decision.get("ok") is False
            and record.decision.get("error", {}).get("error_type") == "CapabilityDenied"
            for record in runtime.store.list_audit()
        )

        exited = call("process_exit", result_oid=None, payload={"summary": "Verified"})
        assert exited["ok"], exited
        assert exited["payload"]["status"] == "exited"
        assert exited["payload"]["terminal_committed"] is True
        assert runtime.process.get(pid).status == ProcessStatus.EXITED
        assert called >= set(REQUIRED_FIELD_CONTRACTS)
    finally:
        runtime.close()
