from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel

from agent_libos.llm.actions import _model_facing_error
from agent_libos.models import ToolCallResult
from agent_libos.tools.base import (
    SyncAgentTool,
    ToolContext,
    ToolErrorCode,
    ToolPolicy,
    ToolResult,
)
from agent_libos.tools.builtin.capabilities import DelegateCapabilityTool
from agent_libos.tools.builtin.checkpoint import RestoreCheckpointTool
from agent_libos.tools.builtin.git import GitDiffTool
from agent_libos.tools.builtin.images import CommitCheckpointToImageTool, LoadImagePackageTool
from agent_libos.tools.builtin.process import ProcessExitTool
from agent_libos.tools.builtin.memory import CreateMemoryObjectTool
from agent_libos.tools.builtin.object_tasks import StartObjectTaskTool, WatchObjectTaskOwnerTool
from agent_libos.tools.builtin.permission import RequestPermissionTool
from agent_libos.tools.observability import json_bytes, sanitize_for_observability


class EmptyArgs(BaseModel):
    pass


class MetadataOnlyTool(SyncAgentTool[EmptyArgs]):
    name = "metadata_only_tool"
    description = "Exercise ToolPolicy metadata semantics."
    args_schema = EmptyArgs
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_confirmation_required=True,
        declared_permissions={"filesystem.write"},
    )

    def run(self, args: EmptyArgs, ctx: ToolContext) -> dict[str, Any]:
        return {"ok": True, "pid": ctx.pid}


class ManyIntsArgs(BaseModel):
    values: tuple[int, ...]


class ManyIntsTool(SyncAgentTool[ManyIntsArgs]):
    name = "many_ints"
    description = "Exercise bounded projection of many Pydantic errors."
    args_schema = ManyIntsArgs

    def run(self, args: ManyIntsArgs, ctx: ToolContext) -> dict[str, int]:
        return {"count": len(args.values)}


class TestToolPolicyAndObservability:
    def test_tool_policy_is_metadata_not_self_granted_authority(self) -> None:
        tool = MetadataOnlyTool()
        result = tool.invoke(
            {},
            ToolContext(trace_id="trace", call_id="call", pid="pid_test", metadata={}),
        )
        spec = tool.spec()

        assert result.ok
        assert result.data == {"ok": True, "pid": "pid_test"}
        assert spec.required_capabilities == []
        assert spec.policy["declared_permissions"] == {"filesystem.write"}
        assert spec.policy["declared_confirmation_required"] is True

    def test_argument_validation_errors_remain_json_serializable(self) -> None:
        result = GitDiffTool().invoke(
            {
                "scope": "worktree",
                "paths": [
                    {
                        "path": "src/example.py",
                        "path_b64": "c3JjL2V4YW1wbGUucHk=",
                    }
                ],
            },
            ToolContext(
                trace_id="trace",
                call_id="call",
                pid="pid_test",
                metadata={},
            ),
        )

        encoded = json.dumps(result.model_dump(mode="json"))

        assert result.ok is False
        assert result.error is not None
        assert result.error.code.value == "validation_error"
        assert "exactly one of path and path_b64" in encoded
        assert "ValueError(" not in encoded

    def test_failure_model_projection_is_bounded_and_drops_dynamic_telemetry(self) -> None:
        result = ToolResult.failure(
            code=ToolErrorCode.VALIDATION_ERROR,
            message=(
                "invalid input at /Users/alice/work/repo/secret.py\n"
                "Traceback (most recent call last):\n"
                '  File "/Users/alice/work/repo/secret.py", line 42'
            ),
            details={
                "error_type": "PydanticValidationError",
                "errors": [
                    {
                        "loc": ["items", index],
                        "type": "value_error",
                        "msg": "invalid item " + ("x" * 1_000),
                        "input": "must-not-be-copied",
                    }
                    for index in range(1_000)
                ],
            },
            metadata={
                "trace_id": "trace-dynamic",
                "call_id": "call-dynamic",
                "duration_ms": 123.4,
                "materialization_id": "materialization-dynamic",
                "data_flow_context": {
                    "labels": {
                        "sensitivity": "confidential",
                        "trust_level": "untrusted",
                    },
                    "source_refs": [
                        {"oid": f"obj-{index}", "version": 1}
                        for index in range(40)
                    ],
                    "materialization_id": "flow-materialization-dynamic",
                },
            },
        )

        projection = result.model_projection(limit_bytes=8_192)
        encoded = json_bytes(projection)

        assert len(encoded) <= 8_192
        assert projection["error"]["code"] == "validation_error"
        assert projection["error"]["type"] == "PydanticValidationError"
        assert projection["error"]["retryable"] is False
        assert projection["error"]["total_errors"] == 1_000
        assert projection["error"]["omitted"] >= 992
        assert len(projection["error"]["error_hash"]) == 64
        assert projection["data_flow_context"]["source_ref_count"] == 40
        assert projection["data_flow_context"]["labels"]["sensitivity"] == "confidential"
        assert b"/Users/alice" not in encoded
        assert b"Traceback" not in encoded
        assert b"trace-dynamic" not in encoded
        assert b"call-dynamic" not in encoded
        assert b"duration_ms" not in encoded
        assert b"materialization" not in encoded
        assert b"must-not-be-copied" not in encoded

    def test_failure_model_projection_degrades_with_tiny_carrier(self) -> None:
        result = ToolResult.failure(
            code=ToolErrorCode.EXECUTION_ERROR,
            message="错误" * 50_000,
            retryable=True,
            details={"error_type": "SandboxError", "stack": "y" * 100_000},
        )

        projection = result.model_projection(limit_bytes=256)
        encoded = json_bytes(projection)

        assert len(encoded) <= 256
        assert projection["error"]["code"] == "execution_error"
        assert projection["error"]["type"] == "SandboxError"
        assert projection["error"]["retryable"] is True
        assert projection["error"]["omitted"] >= 1
        assert len(projection["error"]["error_hash"]) == 64

    def test_many_pydantic_errors_are_counted_and_sampled(self) -> None:
        result = ManyIntsTool().invoke(
            {"values": ["not-an-integer"] * 1_000},
            ToolContext(
                trace_id="trace",
                call_id="call",
                pid="pid_test",
                metadata={},
            ),
        )

        projection = result.model_projection(limit_bytes=4_096)
        encoded = json_bytes(projection)

        assert len(encoded) <= 4_096
        assert projection["error"]["type"] == "ValidationError"
        assert projection["error"]["total_errors"] == 1_000
        assert projection["error"]["omitted"] >= 992
        assert len(projection["error"]["errors"]) <= 8

    def test_success_model_projection_elides_only_equivalent_content(self) -> None:
        duplicated = ToolResult.success(content='{"value":1}', data={"value": 1})
        distinct = ToolResult.success(content="human explanation", data={"value": 1})

        assert duplicated.model_projection(limit_bytes=1_024) == {"value": 1}
        assert distinct.model_projection(limit_bytes=1_024) == {
            "result": {"value": 1},
            "content": "human explanation",
        }

    def test_model_dispatch_uses_safe_error_and_keeps_host_diagnostic_private(self) -> None:
        result = ToolCallResult(
            call_id="call",
            tool_id="tool",
            result_handle=None,
            payload={
                "ok": False,
                "error": {"safe_message": "Schema validation failed at [redacted]."},
            },
            ok=False,
            error="Schema validation failed for SECRET_HOST_DIAGNOSTIC.",
        )

        assert _model_facing_error(result) == "Schema validation failed at [redacted]."

    def test_success_model_projection_can_differ_from_durable_data(self) -> None:
        result = ToolResult.success(
            data={"value": 1, "metadata": {"source_oids": ["obj_1"]}},
            model_data={"value": 1},
        )

        assert result.model_projection(limit_bytes=1_024) == {"value": 1}
        assert result.model_dump(mode="json")["data"]["metadata"] == {
            "source_oids": ["obj_1"]
        }
        assert "model_data" not in result.model_dump(mode="json")

    def test_create_memory_object_declares_object_write_side_effect(self) -> None:
        spec = CreateMemoryObjectTool().spec()

        assert spec.side_effects == ["object.write"]
        assert spec.policy["side_effects"] is True
        assert spec.policy["declared_permissions"] == {"object.write"}

    def test_start_object_task_declares_object_and_process_side_effects(self) -> None:
        spec = StartObjectTaskTool().spec()

        assert spec.policy["side_effects"] is True
        assert set(spec.side_effects) == {"object.link", "object.write", "process.message", "process.spawn", "tool.call"}

    def test_watch_object_task_owner_declares_message_side_effects(self) -> None:
        spec = WatchObjectTaskOwnerTool().spec()

        assert spec.policy["side_effects"] is True
        assert set(spec.side_effects) == {"object.write", "process.message"}

    def test_request_permission_declares_human_and_capability_side_effects(self) -> None:
        spec = RequestPermissionTool().spec()

        assert spec.policy["side_effects"] is True
        assert set(spec.side_effects) == {"capability.write", "human.ask"}

    def test_authority_and_lifecycle_tools_declare_side_effects(self) -> None:
        process_exit = ProcessExitTool().spec()
        delegate = DelegateCapabilityTool().spec()
        restore = RestoreCheckpointTool().spec()
        load_image_package = LoadImagePackageTool().spec()
        commit_checkpoint = CommitCheckpointToImageTool().spec()

        assert process_exit.policy["side_effects"] is True
        assert "process.lifecycle" in process_exit.side_effects
        assert "capability.write" in delegate.side_effects
        assert "checkpoint.restore" in restore.side_effects
        assert {"image.write", "image.admin"} <= restore.policy["declared_permissions"]
        assert {"image.write", "image.admin"} <= load_image_package.policy["declared_permissions"]
        assert {"image.write", "image.admin"} <= commit_checkpoint.policy["declared_permissions"]
        assert "exact image admin" in restore.description
        assert "replace=true requires exact image admin" in load_image_package.description
        assert "replace=true requires exact image admin" in commit_checkpoint.description

    def test_observability_sanitizes_sensitive_fields_without_hashing_secret_values(self) -> None:
        secret = "SECRET_TOKEN_SHOULD_NOT_APPEAR"
        value = {
            "path": "notes.txt",
            "content": secret,
            "payload": {"nested": secret},
            "metadata": {"source_code": secret, "tests": [{"body": secret}]},
        }

        first = sanitize_for_observability(value)
        second = sanitize_for_observability(value)

        assert first["sha256"] == second["sha256"]
        assert first["bytes"] == second["bytes"]
        assert first["sha256"] != hashlib.sha256(json_bytes(value)).hexdigest()
        assert secret not in first["preview"]
        assert first["redacted"] is True
        assert "sha256" not in first["preview"]

    def test_observability_redacts_common_credential_keys(self) -> None:
        secret = "sk_live_very_secret"
        value = {
            "api_key": secret,
            "accessToken": secret,
            "Authorization": f"Bearer {secret}",
            "nested": {"database-password": secret},
        }

        sanitized = sanitize_for_observability(value, preview_chars=10_000)

        assert secret not in sanitized["preview"]
        assert sanitized["redacted"] is True

    def test_observability_redacts_scalar_credential_patterns(self) -> None:
        secret = "sk_live_scalar_secret"
        quoted_password = "hunter2"
        quoted_token = "plainquotedtoken"
        sentinel = "SECRET_TOKEN_SHOULD_NOT_APPEAR"
        sanitized = sanitize_for_observability(
            f"provider failed Authorization: Bearer {secret}; token={secret}; password='{quoted_password}'; api_key=\"{quoted_token}\"; {sentinel}",
            preview_chars=10_000,
        )
        plain = sanitize_for_observability("ordinary failure message", preview_chars=10_000)

        assert secret not in sanitized["preview"]
        assert quoted_password not in sanitized["preview"]
        assert quoted_token not in sanitized["preview"]
        assert sentinel not in sanitized["preview"]
        assert sanitized["redacted"] is True
        assert plain["redacted"] is False
        assert "ordinary failure message" in plain["preview"]

    def test_observability_redacts_uri_userinfo_cookies_and_credential_metadata(self) -> None:
        dsn_username = "observability-user"
        dsn_password = "observability-dsn-password"
        cookie_secret = "observability-cookie-secret"
        metadata_secret = "observability-metadata-secret"
        connection_secret = "observability-connection-secret"
        value = {
            "diagnostic": (
                "connect failed for "
                f"postgresql+psycopg://{dsn_username}:{dsn_password}@db.example/runtime"
            ),
            "wire_error": f"Cookie: session={cookie_secret}; Secure\nretry scheduled",
            "credentialMetadata": {"opaque": metadata_secret},
            "connectionString": f"Server=db.example;Password={connection_secret}",
            "endpoint": "https://api.example.test/v1/health?verbose=true",
            "note": "ordinary provider failure",
        }

        sanitized = sanitize_for_observability(value, preview_chars=10_000)
        preview = sanitized["preview"]

        for secret in (
            dsn_username,
            dsn_password,
            cookie_secret,
            metadata_secret,
            connection_secret,
        ):
            assert secret not in preview
        assert "postgresql+psycopg://[redacted]@db.example/runtime" in preview
        assert "Cookie: [redacted]\\nretry scheduled" in preview
        assert "https://api.example.test/v1/health?verbose=true" in preview
        assert "ordinary provider failure" in preview
        assert sanitized["redacted"] is True
        assert sanitized["sha256"] != hashlib.sha256(json_bytes(value)).hexdigest()

    def test_observability_preserves_noncredential_urls_and_structured_diagnostics(self) -> None:
        value = {
            "endpoint": "https://api.example.test/v1/health?verbose=true",
            "status": 503,
            "retryable": True,
            "note": "ordinary provider failure",
        }

        sanitized = sanitize_for_observability(value, preview_chars=10_000)

        assert sanitized["redacted"] is False
        assert sanitized["preview"] == json.dumps(value, sort_keys=True)
