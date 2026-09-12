from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from agent_libos import Runtime
from agent_libos.config import ProviderToolsConfig
from agent_libos.llm.replay import ReplayStateError
from agent_libos.models import (
    CapabilityRight, SinkTrustLevel, TaskRunRetention, TaskRunSpecV1, TaskRunStatus,
)
from tests.runtime.test_provider_tools_executor import (
    _IMAGE, _Responses, _capture, _config, _spawn,
)
from tests.security.test_responses_replay_security import _sink, _source_view


def _hosted_config(provider: str):
    config = _config()
    return replace(
        config,
        llm=replace(config.llm, profiles={
            "default": replace(
                config.llm.profiles["default"],
                provider_tools=ProviderToolsConfig(provider=provider, web_search=True),
            ),
        }),
        task_runs=replace(config.task_runs, plaintext_payloads_enabled=True),
    )


@pytest.mark.parametrize("provider", ["openai", "aliyun"])
def test_hosted_replay_release_reopens_and_dispatches_frozen_input_once(
    tmp_path: Path, provider: str,
) -> None:
    config = _hosted_config(provider)
    database = tmp_path / "hosted-release.sqlite"
    runtime = Runtime.open(database, config=config)
    try:
        endpoint = _Responses("action")
        _capture(runtime, endpoint)
        _sink(runtime, SinkTrustLevel.CONDITIONAL)
        pid = _spawn(runtime)
        source = _source_view(runtime, pid, secret=True)

        waiting = runtime.run_process_once(pid)

        assert waiting["waiting_human"], waiting
        assert endpoint.requests == []
        pending = runtime.store.get_llm_pending_action(pid)
        reference = pending["action"]["responses_replay_request"]
        frozen = runtime.store.get_llm_replay_turn(reference["turn_id"])
        assert frozen.payload["request"]["payload"]["provider"] == provider
        frozen_input = frozen.payload["request"]["response_items"]
    finally:
        runtime.close()

    reopened = Runtime.open(database, config=config)
    try:
        endpoint = _Responses("action")
        _capture(reopened, endpoint)
        assert reopened.store.get_llm_pending_action(pid)["request_id"] == pending["request_id"]
        assert reopened.capability.check(pid, f"object:{source.oid}", CapabilityRight.READ)
        reopened.human.drain_terminal_queue(auto_approve=True)

        resumed = reopened.run_process_once(pid)

        assert resumed["ok"] and resumed["resumed_after_human"], resumed
        assert len(endpoint.requests) == 1
        assert endpoint.requests[0]["input"] == frozen_input
        assert reopened.store.get_llm_pending_action(pid)["status"] == "completed"
    finally:
        reopened.close()


@pytest.mark.parametrize("provider", ["openai", "aliyun"])
def test_taskrun_hosted_release_preflight_validates_persisted_provider_binding(
    tmp_path: Path, provider: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open(tmp_path / "durable-hosted-release.sqlite", config=_hosted_config(provider))
    try:
        endpoint = _Responses("action")
        _capture(runtime, endpoint)
        _sink(runtime, SinkTrustLevel.CONDITIONAL)
        _spawn(runtime)
        created = runtime.task_runs.create(
            TaskRunSpecV1(
                goal="Research the retained source.", display_title="Hosted release",
                image_id=_IMAGE, retention=TaskRunRetention.PERMANENT,
            ),
            client_request_id="create-hosted-release",
        )
        pid = created.root_pid
        _source_view(runtime, pid, secret=True)
        waiting = runtime.task_runs.run_until_blocked(
            created.run_id, expected_revision=created.revision,
            command_id="wait-for-release", max_quanta=1,
        )
        assert waiting.status is TaskRunStatus.WAITING_HUMAN, waiting.blockers
        assert endpoint.requests == []
        record = runtime.store.get_task_run(created.run_id)

        def reject_live_lookup(*_args):
            raise AssertionError("early payload preflight must not resolve live profiles")

        monkeypatch.setattr(runtime.llms, "profile_snapshot", reject_live_lookup)
        runtime.task_runs._prevalidate_recoverable_record(record)

        pending = deepcopy(runtime.store.get_llm_pending_action(pid))
        pending["action"]["request_options"]["provider_tools_configured"]["provider"] = (
            "aliyun" if provider == "openai" else "openai"
        )
        monkeypatch.setattr(runtime.store, "get_llm_pending_action", lambda _pid: pending)
        with pytest.raises(ReplayStateError, match="frozen request provider changed"):
            runtime.task_runs._prevalidate_recoverable_record(record)
        assert runtime.store.get_task_run(created.run_id) == record
        assert endpoint.requests == []
    finally:
        runtime.close()
