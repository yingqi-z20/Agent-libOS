from __future__ import annotations

import json
import tempfile
import time
from dataclasses import replace

import pytest
from pydantic import BaseModel

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import EventType, ObjectType, ProcessStatus
from agent_libos.tools.base import SyncAgentTool, ToolContext, ToolPolicy


class EmptyArgs(BaseModel):
    pass


class HugeResultTool(SyncAgentTool[EmptyArgs]):
    name = "huge_result"
    description = "Return a result larger than the broker persistence boundary."
    args_schema = EmptyArgs

    def run(self, args: EmptyArgs, ctx: ToolContext) -> dict[str, str]:
        assert ctx.runtime is not None
        return {"blob": "x" * (ctx.runtime.config.tools.tool_result_payload_hard_limit_bytes + 1)}


class HugeSideEffectResultTool(SyncAgentTool[EmptyArgs]):
    name = "huge_side_effect_result"
    description = "Mutate runtime state and then return a result larger than the broker persistence boundary."
    args_schema = EmptyArgs
    policy = ToolPolicy(side_effects=True, idempotent=False, declared_permissions={"object.write"})

    def run(self, args: EmptyArgs, ctx: ToolContext) -> dict[str, str]:
        assert ctx.runtime is not None
        ctx.runtime.memory.create_object(
            ctx.pid,
            ObjectType.OBSERVATION,
            {"committed": True},
            name="huge.side.effect.committed",
        )
        return {"blob": "x" * (ctx.runtime.config.tools.tool_result_payload_hard_limit_bytes + 1)}


class MediumResultTool(SyncAgentTool[EmptyArgs]):
    name = "medium_result"
    description = "Return a result that fits the broker boundary but not Object Memory."
    args_schema = EmptyArgs

    def run(self, args: EmptyArgs, ctx: ToolContext) -> dict[str, str]:
        assert ctx.runtime is not None
        return {"blob": "x" * (ctx.runtime.config.tools.memory_payload_hard_limit_bytes + 1)}


class MediumSideEffectResultTool(SyncAgentTool[EmptyArgs]):
    name = "medium_side_effect_result"
    description = "Mutate runtime state then return a result that is too large to persist in Object Memory."
    args_schema = EmptyArgs
    policy = ToolPolicy(side_effects=True, idempotent=False, declared_permissions={"object.write"})

    def run(self, args: EmptyArgs, ctx: ToolContext) -> dict[str, str]:
        assert ctx.runtime is not None
        ctx.runtime.memory.create_object(
            ctx.pid,
            ObjectType.OBSERVATION,
            {"committed": True},
            name="medium.side.effect.committed",
        )
        return {"blob": "x" * (ctx.runtime.config.tools.memory_payload_hard_limit_bytes + 1)}


class SlowSyncSideEffectTool(SyncAgentTool[EmptyArgs]):
    name = "slow_sync_side_effect"
    description = "Sleep past the declared timeout and then mutate runtime state."
    args_schema = EmptyArgs
    policy = ToolPolicy(side_effects=True, timeout_s=0.01)

    def run(self, args: EmptyArgs, ctx: ToolContext) -> dict[str, bool]:
        time.sleep(0.05)
        assert ctx.runtime is not None
        ctx.runtime.memory.create_object(
            ctx.pid,
            ObjectType.OBSERVATION,
            {"late": True},
            name="sync.side.effect.done",
        )
        return {"done": True}


class ExpectedOutput(BaseModel):
    value: int


class BadStaticOutputTool(SyncAgentTool[EmptyArgs]):
    name = "bad_static_output"
    description = "Return data that violates output_schema."
    args_schema = EmptyArgs
    output_schema = ExpectedOutput

    def run(self, args: EmptyArgs, ctx: ToolContext) -> dict[str, str]:
        return {"value": "not an int"}


class StaticToolV1(SyncAgentTool[EmptyArgs]):
    name = "same_static_tool"
    description = "v1 description"
    args_schema = EmptyArgs

    def run(self, args: EmptyArgs, ctx: ToolContext) -> dict[str, int]:
        return {"version": 1}


class StaticToolV2(SyncAgentTool[EmptyArgs]):
    name = "same_static_tool"
    description = "v2 description"
    args_schema = EmptyArgs

    def run(self, args: EmptyArgs, ctx: ToolContext) -> dict[str, int]:
        return {"version": 2}


class TestToolResultLimits:
    def test_terminalization_during_result_persistence_fails_closed_and_is_audited(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="terminal result persistence race",
            )
            runtime.tools.configure_process_tools(
                pid,
                ["get_working_directory"],
                assigned_by="test",
            )
            before_result_oids = {
                obj.oid
                for obj in runtime.store.list_objects()
                if obj.type == ObjectType.TOOL_RESULT
            }
            result_memory = runtime.tools.execution._memory
            original_create_object = result_memory.create_object
            terminalized = False
            terminal_revision: int | None = None
            terminal_capability_ids: tuple[str, ...] = ()

            def terminalize_before_result_create(
                *args: object,
                **kwargs: object,
            ):
                nonlocal terminalized, terminal_revision, terminal_capability_ids
                selected_type = kwargs.get("object_type")
                if selected_type is None and len(args) > 1:
                    selected_type = args[1]
                if selected_type == ObjectType.TOOL_RESULT and not terminalized:
                    terminalized = True
                    runtime.process.cancel(
                        pid,
                        "injected tool-result persistence race",
                    )
                    terminal = runtime.process.get(pid)
                    terminal_revision = terminal.revision
                    terminal_capability_ids = tuple(terminal.capabilities)
                return original_create_object(*args, **kwargs)

            monkeypatch.setattr(
                result_memory,
                "create_object",
                terminalize_before_result_create,
            )

            result = runtime.tools.call(pid, "get_working_directory", {})

            assert terminalized
            assert not result.ok
            assert result.result_handle is None
            assert "terminal process" in (result.error or "")
            terminal = runtime.process.get(pid)
            assert terminal.status == ProcessStatus.KILLED
            assert terminal.revision == terminal_revision
            assert tuple(terminal.capabilities) == terminal_capability_ids
            after_result_oids = {
                obj.oid
                for obj in runtime.store.list_objects()
                if obj.type == ObjectType.TOOL_RESULT
            }
            assert after_result_oids == before_result_oids
            call_events = [
                event
                for event in runtime.events.list()
                if event.payload.get("call_id") == result.call_id
                and event.type in {EventType.TOOL_COMPLETED, EventType.TOOL_FAILED}
            ]
            assert [event.type for event in call_events] == [EventType.TOOL_FAILED]
            assert call_events[0].payload["result_oid"] is None
            tool_audits = [
                record
                for record in runtime.audit.trace()
                if record.action == "tool.call"
                and record.decision.get("tool") == "get_working_directory"
            ]
            assert len(tool_audits) == 1
            assert tool_audits[0].decision["ok"] is False
            assert tool_audits[0].decision["policy_decision"] == "allow"
            assert tool_audits[0].output_refs == []
        finally:
            runtime.close()

    def test_tool_result_payload_limit_rejects_before_result_object_creation(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="huge result")
            handle = runtime.tools.register_tool(HugeResultTool(), registered_by="test", ephemeral=True)
            runtime.tools.configure_process_tools(pid, [handle], assigned_by="test")

            result = runtime.tools.call(pid, "huge_result", {})

            assert not result.ok
            assert result.result_handle is None
            assert "tool result payload exceeds" in (result.error or "")
            assert [obj for obj in runtime.store.list_objects() if obj.type.value == "tool_result"] == []
            audit = [record for record in runtime.audit.trace() if record.action == "tool.call"][-1]
            assert audit.decision["result"]["preview"] == "[tool result omitted after size-limit failure]"
        finally:
            runtime.close()

    def test_side_effect_tool_oversize_result_is_reported_as_omitted_success(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="huge side effect result")
            handle = runtime.tools.register_tool(HugeSideEffectResultTool(), registered_by="test", ephemeral=True)
            runtime.tools.configure_process_tools(pid, [handle], assigned_by="test")

            result = runtime.tools.call(pid, "huge_side_effect_result", {})

            assert result.ok, result.error
            assert result.result_handle is not None
            assert result.payload["result_omitted"] is True
            assert runtime.store.get_object_by_name(
                "huge.side.effect.committed",
                namespace=runtime.memory.resolve_namespace(pid),
            ) is not None
            stored = runtime.store.get_object(result.result_handle.oid)
            assert stored is not None
            assert stored.payload["metadata"]["result_omitted"] is True
            audit = [record for record in runtime.audit.trace() if record.action == "tool.call"][-1]
            assert audit.decision["ok"] is True
            assert audit.decision["result_omitted"] is True
        finally:
            runtime.close()

    def test_result_too_large_for_object_memory_is_rejected_without_result_object(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="medium result")
            handle = runtime.tools.register_tool(MediumResultTool(), registered_by="test", ephemeral=True)
            runtime.tools.configure_process_tools(pid, [handle], assigned_by="test")

            result = runtime.tools.call(pid, "medium_result", {})

            assert not result.ok
            assert result.result_handle is None
            assert "tool result payload exceeds" in (result.error or "")
            assert [obj for obj in runtime.store.list_objects() if obj.type.value == "tool_result"] == []
        finally:
            runtime.close()

    def test_side_effect_result_too_large_for_object_memory_is_reported_as_omitted_success(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="medium side effect result")
            handle = runtime.tools.register_tool(MediumSideEffectResultTool(), registered_by="test", ephemeral=True)
            runtime.tools.configure_process_tools(pid, [handle], assigned_by="test")

            result = runtime.tools.call(pid, "medium_side_effect_result", {})

            assert result.ok, result.error
            assert result.result_handle is not None
            assert result.payload["result_omitted"] is True
            assert runtime.store.get_object_by_name(
                "medium.side.effect.committed",
                namespace=runtime.memory.resolve_namespace(pid),
            ) is not None
            stored = runtime.store.get_object(result.result_handle.oid)
            assert stored is not None
            assert stored.payload["metadata"]["result_omitted"] is True
        finally:
            runtime.close()

    def test_sync_side_effect_tool_does_not_return_before_background_work_finishes(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="sync timeout")
            handle = runtime.tools.register_tool(SlowSyncSideEffectTool(), registered_by="test", ephemeral=True)
            runtime.tools.configure_process_tools(pid, [handle], assigned_by="test")

            result = runtime.tools.call(pid, "slow_sync_side_effect", {})

            assert result.ok, result.error
            assert runtime.store.get_object_by_name(
                "sync.side.effect.done",
                namespace=runtime.memory.resolve_namespace(pid),
            ) is not None
        finally:
            runtime.close()

    def test_static_tool_output_schema_is_enforced(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="bad output")
            handle = runtime.tools.register_tool(BadStaticOutputTool(), registered_by="test", ephemeral=True)
            runtime.tools.configure_process_tools(pid, [handle], assigned_by="test")

            result = runtime.tools.call(pid, "bad_static_output", {})

            assert not result.ok
            assert "Invalid output" in (result.error or "")
            assert result.result_handle is None
        finally:
            runtime.close()

    def test_static_tool_spec_is_refreshed_when_runtime_reopens_with_new_implementation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = f"{temp_dir}/runtime.sqlite"
            runtime = Runtime.open(db_path)
            try:
                runtime.tools.register_tool(StaticToolV1(), registered_by="test")
            finally:
                runtime.close()

            runtime = Runtime.open(db_path)
            try:
                runtime.tools.register_tool(StaticToolV2(), registered_by="test")
                row = next(row for row in runtime.store.list_tools() if row["name"] == "same_static_tool")
                spec = json.loads(row["spec_json"])
                assert spec["description"] == "v2 description"
            finally:
                runtime.close()

    def test_oversize_tool_arguments_are_rejected_before_observability_event(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            tools=replace(DEFAULT_CONFIG.tools, tool_call_args_hard_limit_bytes=256),
        )
        runtime = Runtime.open("local", config=config)
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="oversize args")
            runtime.tools.configure_process_tools(pid, ["echo"], assigned_by="test")

            result = runtime.tools.call(pid, "echo", {"blob": "x" * 1000})

            assert not result.ok
            assert result.result_handle is None
            assert "tool call arguments exceed" in (result.error or "")
            assert all(event.type.value != "tool_called" for event in runtime.events.list(target=pid))
        finally:
            runtime.close()
