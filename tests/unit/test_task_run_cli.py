from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import create_autospec

import pytest

import agent_libos.models as models
from agent_libos.api.cli import _parse_cli_args, _run_task_run_command
from agent_libos.models.exceptions import ValidationError


@dataclass(frozen=True)
class _FakeTaskRunSpecV1:
    schema_version: int
    goal: Any
    display_title: str
    image_id: str | None
    launch_options: dict[str, Any]
    authority_manifest_id: str | None
    deadline_at: str | None
    retention: str


class _FakeTaskRuns:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _record(self, name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((name, args, kwargs))
        return {
            "run_id": "run-1",
            "revision": len(self.calls),
            "status": "waiting_human",
            "allowed_actions": ["follow_up", "cancel"],
        }

    def create(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._record("create", *args, **kwargs)

    def get(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._record("get", *args, **kwargs)

    def list(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("list", args, kwargs))
        return {"items": [], "next_cursor": None, "has_more": False}

    def run_until_blocked(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._record("run_until_blocked", *args, **kwargs)

    def wait(
        self,
        run_id: str,
        *,
        after_revision: int | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self._record(
            "wait",
            run_id,
            after_revision=after_revision,
            timeout=timeout,
        )

    def recovery_options(self, *args: Any, **kwargs: Any) -> tuple[dict[str, Any], ...]:
        self.calls.append(("recovery_options", args, kwargs))
        return (
            {
                "option_id": "confirmed_not_started:binding-1",
                "kind": "confirmed_not_started",
                "label": "Confirm provider dispatch did not start",
                "requires_receipt": False,
            },
        )

    def pause(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._record("pause", *args, **kwargs)

    def resume(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._record("resume", *args, **kwargs)

    def cancel(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._record("cancel", *args, **kwargs)

    def follow_up(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._record("follow_up", *args, **kwargs)

    def recover(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._record("recover", *args, **kwargs)

    def rerun(
        self,
        source_run_id: str,
        *,
        expected_revision: int,
        command_id: str,
        client_request_id: str | None,
        spec_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._record(
            "rerun",
            source_run_id,
            expected_revision=expected_revision,
            command_id=command_id,
            client_request_id=client_request_id,
            spec_overrides=spec_overrides,
        )


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(
        task_runs=_FakeTaskRuns(),
        config=SimpleNamespace(task_runs=SimpleNamespace(payload_max_bytes=16_384)),
    )


def _args(argv: list[str]) -> Any:
    return _parse_cli_args(["task-run", *argv])[1]


def test_task_run_start_is_queued_by_default_and_uses_canonical_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models, "TaskRunSpecV1", _FakeTaskRunSpecV1, raising=False)
    runtime = _runtime()

    result = _run_task_run_command(
        runtime,
        _args(
            [
                "start",
                "--goal-json",
                '{"ticket":"T-1"}',
                "--title",
                "Repair T-1",
                "--launch-json",
                '{"working_directory":"repo"}',
                "--authority-manifest-id",
                "authority-1",
                "--deadline-at",
                "2030-01-01T00:00:00Z",
                "--retention",
                "permanent",
                "--client-request-id",
                "client-1",
            ]
        ),
    )

    assert result["status"] == "waiting_human"
    assert [call[0] for call in runtime.task_runs.calls] == ["create"]
    spec = runtime.task_runs.calls[0][1][0]
    assert spec == _FakeTaskRunSpecV1(
        schema_version=1,
        goal={"ticket": "T-1"},
        display_title="Repair T-1",
        image_id=None,
        launch_options={"working_directory": "repo"},
        authority_manifest_id="authority-1",
        deadline_at="2030-01-01T00:00:00Z",
        retention="permanent",
    )
    assert runtime.task_runs.calls[0][2] == {
        "client_request_id": "client-1",
        "auto_run": False,
    }


def test_task_run_start_run_is_a_separate_revision_fenced_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models, "TaskRunSpecV1", _FakeTaskRunSpecV1, raising=False)
    runtime = _runtime()

    result = _run_task_run_command(
        runtime,
        _args(
            [
                "start",
                "--goal",
                "continue until blocked",
                "--title",
                "Durable work",
                "--run",
                "--run-command-id",
                "run-2",
                "--max-quanta",
                "7",
            ]
        ),
    )

    assert result["status"] == "waiting_human"
    assert [call[0] for call in runtime.task_runs.calls] == [
        "create",
        "run_until_blocked",
    ]
    assert runtime.task_runs.calls[1] == (
        "run_until_blocked",
        ("run-1",),
        {
            "expected_revision": 1,
            "command_id": "run-2",
            "max_quanta": 7,
        },
    )


@pytest.mark.parametrize("max_quanta", ["0", "-1"])
def test_task_run_start_run_rejects_nonpositive_quanta_before_create(
    monkeypatch: pytest.MonkeyPatch,
    max_quanta: str,
) -> None:
    monkeypatch.setattr(models, "TaskRunSpecV1", _FakeTaskRunSpecV1, raising=False)
    runtime = _runtime()

    with pytest.raises(
        ValidationError,
        match="max_quanta must be a positive integer",
    ):
        _run_task_run_command(
            runtime,
            _args(
                [
                    "start",
                    "--goal",
                    "must not be created",
                    "--title",
                    "Rejected quantum budget",
                    "--run",
                    "--max-quanta",
                    max_quanta,
                ]
            ),
        )

    assert runtime.task_runs.calls == []


@pytest.mark.parametrize("command", ["cancel", "recover"])
def test_task_run_destructive_commands_require_explicit_confirmation(
    command: str,
) -> None:
    argv = ["task-run", command, "run-1", "--expected-revision", "1"]
    if command == "recover":
        argv.append("confirmed_not_started")

    with pytest.raises(SystemExit) as raised:
        _parse_cli_args(argv)

    assert raised.value.code == 2


def test_task_run_waiting_and_needs_attention_are_returned_as_normal_summaries() -> None:
    runtime = _runtime()

    waiting = _run_task_run_command(
        runtime,
        _args(["wait", "run-1", "--timeout", "0"]),
    )
    runtime.task_runs._record = lambda *_args, **_kwargs: {
        "run_id": "run-1",
        "revision": 6,
        "status": "needs_attention",
        "blockers": [{"kind": "unknown_effect"}],
        "allowed_actions": ["recover", "cancel"],
    }
    attention = _run_task_run_command(
        runtime,
        _args(
            [
                "recover",
                "run-1",
                "register_authoritative_receipt",
                "--expected-revision",
                "5",
                "--command-id",
                "recover-1",
                "--receipt-json",
                '{"receipt_id":"receipt-1"}',
                "--confirm",
            ]
        ),
    )

    assert waiting["status"] == "waiting_human"
    assert runtime.task_runs.calls[0] == (
        "wait",
        ("run-1",),
        {"after_revision": None, "timeout": 0.0},
    )
    assert attention["status"] == "needs_attention"
    assert attention["blockers"] == [{"kind": "unknown_effect"}]


def test_task_run_recovery_options_is_a_read_only_server_derived_projection() -> None:
    runtime = _runtime()

    result = _run_task_run_command(
        runtime,
        _args(["recovery-options", "run-1"]),
    )

    assert result == [
        {
            "option_id": "confirmed_not_started:binding-1",
            "kind": "confirmed_not_started",
            "label": "Confirm provider dispatch did not start",
            "requires_receipt": False,
        }
    ]
    assert runtime.task_runs.calls == [("recovery_options", ("run-1",), {})]


def test_task_run_parser_exposes_the_exact_supported_subcommands() -> None:
    parser, _args_namespace = _parse_cli_args(["task-run", "get", "run-1"])

    task_run_action = next(
        action
        for action in parser._actions
        if getattr(action, "dest", None) == "command"
    )
    task_run_parser = task_run_action.choices["task-run"]
    command_action = next(
        action
        for action in task_run_parser._actions
        if getattr(action, "dest", None) == "task_run_command"
    )

    assert tuple(command_action.choices) == (
        "start",
        "get",
        "list",
        "wait",
        "recovery-options",
        "pause",
        "resume",
        "cancel",
        "follow-up",
        "recover",
        "rerun",
    )


def test_task_run_list_status_filter_is_bound_to_the_status_enum() -> None:
    # Regression: the closed status-filter set, the --status help text, and
    # the rejection message must all be derived from TaskRunStatus so the
    # valid values are discoverable without reading source code.
    from agent_libos.api.cli import _TASK_RUN_STATUSES, _task_run_status_filters

    assert _TASK_RUN_STATUSES == {status.value for status in models.TaskRunStatus}

    parser, _args_namespace = _parse_cli_args(["task-run", "list"])
    task_run_action = next(
        action
        for action in parser._actions
        if getattr(action, "dest", None) == "command"
    )
    task_run_parser = task_run_action.choices["task-run"]
    command_action = next(
        action
        for action in task_run_parser._actions
        if getattr(action, "dest", None) == "task_run_command"
    )
    list_parser = command_action.choices["list"]
    status_action = next(
        action
        for action in list_parser._actions
        if getattr(action, "dest", None) == "statuses"
    )
    for status in models.TaskRunStatus:
        assert status.value in status_action.help

    with pytest.raises(ValidationError) as excinfo:
        _task_run_status_filters(["queued,not_a_status"])
    assert "not_a_status" in str(excinfo.value)
    for status in models.TaskRunStatus:
        assert status.value in str(excinfo.value)


def test_task_run_list_passes_bounded_status_filters_and_cursor() -> None:
    runtime = _runtime()

    result = _run_task_run_command(
        runtime,
        _args(
            [
                "list",
                "--status",
                "queued,waiting_human",
                "--status",
                "needs_attention",
                "--cursor",
                "cursor-1",
                "--limit",
                "500",
            ]
        ),
    )

    assert result == {"items": [], "next_cursor": None, "has_more": False}
    assert runtime.task_runs.calls == [
        (
            "list",
            (),
            {
                "statuses": ("queued", "waiting_human", "needs_attention"),
                "cursor": "cursor-1",
                "limit": 500,
            },
        )
    ]


def test_task_run_mutations_use_canonical_manager_contract() -> None:
    runtime = _runtime()

    _run_task_run_command(
        runtime,
        _args(
            [
                "follow-up",
                "run-1",
                "check the release",
                "--interrupt",
                "--optional",
                "--expected-revision",
                "3",
                "--command-id",
                "follow-1",
            ]
        ),
    )
    _run_task_run_command(
        runtime,
        _args(
            [
                "recover",
                "run-1",
                "register_receipt",
                "--receipt-json",
                '{"receipt_id":"r-1"}',
                "--expected-revision",
                "4",
                "--command-id",
                "recover-1",
                "--confirm",
            ]
        ),
    )

    assert runtime.task_runs.calls[0] == (
        "follow_up",
        ("run-1",),
        {
            "expected_revision": 3,
            "command_id": "follow-1",
            "body": "check the release",
            "kind": "interrupt",
            "required": False,
        },
    )
    assert runtime.task_runs.calls[1] == (
        "recover",
        ("run-1",),
        {
            "expected_revision": 4,
            "command_id": "recover-1",
            "option_id": "register_receipt",
            "receipt": {"receipt_id": "r-1"},
        },
    )


def test_task_run_rerun_reuses_manager_derived_create_id_across_cli_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    monkeypatch.setattr(
        "agent_libos.api.cli.uuid.uuid4",
        lambda: pytest.fail("stable rerun retry must not generate a client request id"),
    )

    argv = [
        "rerun",
        "run-1",
        "--expected-revision",
        "9",
        "--command-id",
        "stable-rerun-command",
    ]
    _run_task_run_command(runtime, _args(argv))
    _run_task_run_command(runtime, _args(argv))

    assert runtime.task_runs.calls == [
        (
            "rerun",
            ("run-1",),
            {
                "expected_revision": 9,
                "command_id": "stable-rerun-command",
                "client_request_id": None,
                "spec_overrides": None,
            },
        ),
        (
            "rerun",
            ("run-1",),
            {
                "expected_revision": 9,
                "command_id": "stable-rerun-command",
                "client_request_id": None,
                "spec_overrides": None,
            },
        )
    ]


def test_task_run_start_help_names_the_runtime_default_image() -> None:
    parser, _args_namespace = _parse_cli_args(
        ["task-run", "start", "--goal", "x", "--title", "X"]
    )

    task_run_action = next(
        action
        for action in parser._actions
        if getattr(action, "dest", None) == "command"
    )
    task_run_parser = task_run_action.choices["task-run"]
    command_action = next(
        action
        for action in task_run_parser._actions
        if getattr(action, "dest", None) == "task_run_command"
    )
    start_parser = command_action.choices["start"]
    image_action = next(
        action for action in start_parser._actions if action.dest == "image"
    )

    assert "configured default image" in image_action.help
    assert "coding image" not in image_action.help


def test_task_run_cli_calls_are_bound_to_real_manager_signatures() -> None:
    # An autospecced manager makes this test fail immediately when a CLI keyword
    # drifts from the public Runtime.task_runs contract.
    from agent_libos.runtime.task_runs import TaskRunManager

    manager = create_autospec(TaskRunManager, instance=True)
    manager.wait.return_value = {"run_id": "run-1", "revision": 1}
    manager.recovery_options.return_value = ()
    manager.rerun.return_value = {"run_id": "run-2", "revision": 0}
    runtime = SimpleNamespace(
        task_runs=manager,
        config=SimpleNamespace(task_runs=SimpleNamespace(payload_max_bytes=16_384)),
    )

    _run_task_run_command(runtime, _args(["wait", "run-1", "--timeout", "2.5"]))
    _run_task_run_command(runtime, _args(["recovery-options", "run-1"]))
    _run_task_run_command(
        runtime,
        _args(
            [
                "rerun",
                "run-1",
                "--expected-revision",
                "4",
                "--command-id",
                "rerun-command",
                "--client-request-id",
                "rerun-client",
            ]
        ),
    )

    manager.wait.assert_called_once_with("run-1", timeout=2.5)
    manager.recovery_options.assert_called_once_with("run-1")
    manager.rerun.assert_called_once_with(
        "run-1",
        expected_revision=4,
        command_id="rerun-command",
        client_request_id="rerun-client",
        spec_overrides=None,
    )
