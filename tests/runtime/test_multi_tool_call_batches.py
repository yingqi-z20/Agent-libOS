from __future__ import annotations

from pathlib import Path

import pytest

from dataclasses import replace

from agent_libos import Runtime, TaskRunSpecV1
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import TaskRunRetention
from agent_libos.skills.builtin_catalog import get_builtin_skill_catalog

_TASK_RUN_CONFIG = replace(
    DEFAULT_CONFIG,
    task_runs=replace(DEFAULT_CONFIG.task_runs, plaintext_payloads_enabled=True),
)


def _activate_action(skill_id: str) -> dict[str, str]:
    package = get_builtin_skill_catalog().get(skill_id)
    assert package is not None
    return {
        "action": "activate_skill",
        "skill_id": skill_id,
        "expected_package_sha256": package.package_sha256,
    }


def test_durable_task_run_batch_with_activate_skill_is_repaired_not_failed(
    tmp_path: Path,
) -> None:
    """A multi-call response that rebinds tools inside a Durable TaskRun.

    The TaskRun manifest records ``activate_skill`` only as a singleton action.
    The executor must turn that shape into an ordinary action repair before the
    provider completion is committed, instead of failing the quantum after a
    paid completion was already admitted.
    """

    runtime = Runtime.open(tmp_path / "batch-guard.sqlite", config=_TASK_RUN_CONFIG)
    try:
        created = runtime.task_runs.create(
            TaskRunSpecV1(
                goal="guard binding batches",
                display_title="Binding batch guard",
                image_id="coding-agent:v0",
                retention=TaskRunRetention.PERMANENT,
            ),
            client_request_id="create-binding-batch-guard",
        )
        root_pid = created.root_pid
        assert root_pid is not None

        with pytest.raises(ValueError, match="activate_skill only as the single"):
            runtime.llm._validate_multi_action_batch(
                root_pid,
                [
                    _activate_action("agent-libos-workspace-navigation"),
                    {"action": "discover_skills", "text": "workspace", "limit": 5},
                ],
            )
        # Batches without a binding mutation stay dispatchable.
        runtime.llm._validate_multi_action_batch(
            root_pid,
            [
                {"action": "discover_skills", "text": "workspace", "limit": 5},
                {"action": "discover_skills", "text": "git", "limit": 5},
            ],
        )
    finally:
        runtime.close()


def test_plain_process_batch_with_activate_skill_is_allowed(tmp_path: Path) -> None:
    runtime = Runtime.open(tmp_path / "plain-batch.sqlite")
    try:
        pid = runtime.process.spawn(image="coding-agent:v0", goal="plain batch")
        runtime.llm._validate_multi_action_batch(
            pid,
            [
                _activate_action("agent-libos-workspace-navigation"),
                _activate_action("agent-libos-command-execution"),
            ],
        )
    finally:
        runtime.close()
