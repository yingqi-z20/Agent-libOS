from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from agent_libos import Runtime
from agent_libos.models import PROMPT_MODE_IMAGE_ONLY, PROMPT_MODE_LIBOS_DEFAULT
from agent_libos.models.exceptions import ValidationError
from agent_libos.utils.serde import dumps
from tests.runtime.test_provider_tools_executor import (
    _RESULT, _Responses, _assert_local_only, _capture, _config, _spawn,
)


@pytest.mark.parametrize("mode", [PROMPT_MODE_LIBOS_DEFAULT, PROMPT_MODE_IMAGE_ONLY])
@pytest.mark.parametrize("code", [False, True], ids=["search", "code"])
@pytest.mark.parametrize("operation", ["restore", "fork"])
def test_checkpoint_rebinds_pending_local_provider_result_without_repeating_hosted_work(
    mode: str, code: bool, operation: str,
) -> None:
    runtime = Runtime.open("local", config=_config(code=code))
    try:
        provider = _Responses("hosted", "action", "hosted", "action", code=code)
        _capture(runtime, provider)
        pid = _spawn(runtime, mode)
        first = runtime.run_process_once(pid)
        assert first.get("provider_continuation") is True, first
        original = runtime.store.get_latest_llm_call(pid=pid, purpose="provider_continuation")
        checkpoint_id = runtime.checkpoint.create(pid, "pending hosted result", actor=pid)
        snapshot = runtime.store.get_checkpoint_snapshot(checkpoint_id)[1]
        reference = snapshot["provider_continuation_refs"][pid]
        assert reference["marker_call_id"] == original.call_id
        assert reference["source_call_id"] == first["call_id"]
        assert _RESULT not in dumps(reference)
        assert runtime.run_process_once(pid)["ok"]
        discarded = runtime.run_process_once(pid)
        assert discarded.get("provider_continuation") is True
        if operation == "restore":
            runtime.checkpoint.restore(pid, checkpoint_id, require_capability=False)
            target_pid = pid
        else:
            target_pid = runtime.checkpoint.fork_from_checkpoint(
                pid, checkpoint_id, require_capability=False,
            )["fork_root_pid"]
        rebound = runtime.store.get_latest_llm_call(pid=target_pid, purpose="provider_continuation")
        manifest = rebound.request_options["provider_continuation"]
        assert manifest["schema_version"] == 2
        assert manifest["source_pid"] == pid
        assert manifest["call_id"] == first["call_id"] != discarded["call_id"]
        assert manifest["context_generation"] == runtime.store.get_llm_context_generation(target_pid)
        result = runtime.run_process_once(target_pid)
        assert result["ok"], result
        request = provider.requests[-1]
        _assert_local_only(request)
        assert _RESULT in dumps(request["input"])
        if code:
            assert runtime.store.get_llm_replay_head(target_pid) is None
            assert all(item.get("type") != "code_interpreter_call" for item in request["input"])
            assert "container_" not in dumps(request["input"])
            assert request.get("previous_response_id") is None
    finally:
        runtime.close()


@pytest.mark.parametrize("operation", ["restore", "fork"])
def test_checkpoint_predating_hosted_result_does_not_restore_later_marker(operation: str) -> None:
    runtime = Runtime.open("local", config=_config(code=True))
    try:
        provider = _Responses("hosted", "action", code=True)
        _capture(runtime, provider)
        pid = _spawn(runtime)
        checkpoint_id = runtime.checkpoint.create(pid, "before hosted result", actor=pid)
        assert runtime.run_process_once(pid).get("provider_continuation") is True
        if operation == "restore":
            runtime.checkpoint.restore(pid, checkpoint_id, require_capability=False)
            target_pid = pid
        else:
            target_pid = runtime.checkpoint.fork_from_checkpoint(
                pid, checkpoint_id, require_capability=False,
            )["fork_root_pid"]
        result = runtime.run_process_once(target_pid)
        assert result["ok"], result
        assert any(tool["type"] == "code_interpreter" for tool in provider.requests[-1]["tools"])
        assert _RESULT not in dumps(provider.requests[-1]["input"])
    finally:
        runtime.close()


def test_checkpoint_pending_provider_result_fails_closed_when_payload_unavailable() -> None:
    runtime = Runtime.open("local", config=_config(code=True, full_io=False))
    try:
        provider = _Responses("hosted", code=True)
        _capture(runtime, provider)
        pid = _spawn(runtime)
        assert runtime.run_process_once(pid).get("provider_continuation") is True
        with pytest.raises(ValidationError, match="payload retention"):
            runtime.checkpoint.create(pid, "unretained hosted result", actor=pid)
        assert len(provider.requests) == 1
    finally:
        runtime.close()


@pytest.mark.parametrize("tamper", ["marker_sha256", "source_sha256", "source_pid", "payload_sha256"])
def test_checkpoint_continuation_corruption_fails_before_publication(tamper: str) -> None:
    runtime = Runtime.open("local", config=_config(code=True))
    try:
        provider = _Responses("hosted", code=True)
        _capture(runtime, provider)
        pid = _spawn(runtime)
        assert runtime.run_process_once(pid).get("provider_continuation") is True
        checkpoint_id = runtime.checkpoint.create(pid, "retained hosted result", actor=pid)
        snapshot = deepcopy(runtime.store.get_checkpoint_snapshot(checkpoint_id)[1])
        snapshot["provider_continuation_refs"][pid][tamper] = "wrong-pid" if tamper == "source_pid" else "0" * 64
        generation = runtime.store.get_llm_context_generation(pid)
        with pytest.raises(ValidationError, match="provider continuation"):
            runtime.checkpoint._validate_responses_replay(snapshot)
        assert runtime.store.get_llm_context_generation(pid) == generation
        assert len(provider.requests) == 1
    finally:
        runtime.close()


def test_rebound_fork_continuation_retains_local_source_after_reopen(tmp_path: Path) -> None:
    database = tmp_path / "forked-continuation.sqlite"
    config = _config(code=True)
    runtime = Runtime.open(database, config=config)
    try:
        provider = _Responses("hosted", code=True)
        _capture(runtime, provider)
        pid = _spawn(runtime, PROMPT_MODE_IMAGE_ONLY)
        assert runtime.run_process_once(pid).get("provider_continuation") is True
        checkpoint_id = runtime.checkpoint.create(pid, "fork local hosted result", actor=pid)
        target_pid = runtime.checkpoint.fork_from_checkpoint(
            pid, checkpoint_id, require_capability=False,
        )["fork_root_pid"]
    finally:
        runtime.close()
    reopened = Runtime.open(database, config=config)
    try:
        pending = reopened.llm._provider_continuation_data(target_pid)
        assert pending is not None
        assert pending["manifest"]["state"] == "pending"
        assert pending["manifest"]["source_pid"] == pid
        assert _RESULT in pending["message"]["content"]
        assert "container_" not in dumps(pending["message"])
    finally:
        reopened.close()
