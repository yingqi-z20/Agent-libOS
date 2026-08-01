from __future__ import annotations

import http.client
import json
import tempfile
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agent_libos.api.gui.server import create_gui_http_server
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import CapabilityRight
from agent_libos.models.exceptions import (
    TaskRunCommandConflict,
    TaskRunRevisionConflict,
)


def _summary(revision: int = 1, *, status: str = "queued") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": "run-1",
        "revision": revision,
        "status": status,
        "display_title": "Durable repair",
        "root_pid": "pid-root",
        "active_pid": "pid-root",
        "step_count": 3,
        "completed_step_count": 1,
        "requirement_count": 2,
        "satisfied_requirement_count": 1,
        "blockers": [],
        "allowed_actions": ["run", "pause", "cancel", "purge_payloads"],
        "result_ref": None,
        "retention": "purge_on_terminal",
        "payloads_purged": False,
        "created_at": "2030-01-01T00:00:00+00:00",
        "updated_at": "2030-01-01T00:00:01+00:00",
        # These fields must never cross the private summary/SSE boundary.
        "goal": "TASK_RUN_PRIVATE_GOAL_SENTINEL",
        "launch_options": {"api_key": "TASK_RUN_PRIVATE_API_KEY"},
        "provider_secret": "TASK_RUN_PRIVATE_PROVIDER_SECRET",
    }


def _requirement(
    requirement_id: str,
    *,
    retention: str,
    content_text: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 1,
        "requirement_id": requirement_id,
        "run_id": "run-1",
        "ordinal": 0 if requirement_id.endswith("1") else 1,
        "kind": "initial" if requirement_id.endswith("1") else "follow_up",
        "status": "pending",
        "requirement_sha256": "a" * 64,
        "label": "Requirement",
        "created_by": "host",
        "created_at": "2030-01-01T00:00:00+00:00",
        "updated_at": "2030-01-01T00:00:00+00:00",
        "started_at": None,
        "completed_at": None,
        "waived_by": None,
        "content_retention": retention,
        "content_available": retention == "plaintext",
        "content_sha256": "a" * 64,
    }
    if content_text is not None:
        value["content_text"] = content_text
    return value


class _FakeTaskRuns:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.fail_revision = False
        self.fail_command = False
        self.run_started: threading.Event | None = None
        self.cancel_releases_run: threading.Event | None = None
        self.recovery_option_records: list[dict[str, Any]] = []

    def _call(self, name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((name, args, kwargs))
        if self.fail_command:
            raise TaskRunCommandConflict(
                "TaskRun idempotency key was reused with a different request"
            )
        if self.fail_revision:
            raise TaskRunRevisionConflict("stale TaskRun revision")
        return _summary(len(self.calls) + 1)

    def create(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._call("create", *args, **kwargs)

    def get(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._call("get", *args, **kwargs)

    def list(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("list", args, kwargs))
        return {
            "items": [_summary()],
            "next_cursor": "cursor-2",
            "has_more": True,
        }

    def list_ledger(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("list_ledger", args, kwargs))
        return {
            "items": [
                {
                    "schema_version": 1,
                    "item_id": "ledger-1",
                    "run_id": "run-1",
                    "seq": 1,
                    "kind": "status_transition",
                    "status": "queued",
                    "label": "created",
                    "occurred_at": "2030-01-01T00:00:00+00:00",
                    "metadata": {},
                }
            ],
            "next_cursor": None,
            "has_more": False,
        }

    def list_requirements(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("list_requirements", args, kwargs))
        return {
            "items": [
                _requirement(
                    "requirement-1",
                    retention="plaintext",
                    content_text="retained requirement",
                ),
                _requirement("requirement-2", retention="hash_only"),
            ],
            "next_cursor": "requirement-next",
            "has_more": True,
        }

    def recovery_options(self, _run_id: str) -> list[dict[str, Any]]:
        return self.recovery_option_records

    def run_until_blocked(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if self.run_started is not None:
            self.run_started.set()
        if self.cancel_releases_run is not None:
            self.cancel_releases_run.wait(timeout=5)
        return self._call("run_until_blocked", *args, **kwargs)

    def pause(
        self,
        run_id: str,
        *,
        expected_revision: int,
        command_id: str,
    ) -> dict[str, Any]:
        return self._call(
            "pause",
            run_id,
            expected_revision=expected_revision,
            command_id=command_id,
        )

    def resume(
        self,
        run_id: str,
        *,
        expected_revision: int,
        command_id: str,
    ) -> dict[str, Any]:
        return self._call(
            "resume",
            run_id,
            expected_revision=expected_revision,
            command_id=command_id,
        )

    def cancel(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if self.cancel_releases_run is not None:
            self.cancel_releases_run.set()
        return self._call("cancel", *args, **kwargs)

    def follow_up(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._call("follow_up", *args, **kwargs)

    def recover(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._call("recover", *args, **kwargs)

    def rerun(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._call("rerun", *args, **kwargs)


class TestTaskRunGuiAPI:
    def setup_method(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.server = create_gui_http_server(
            db="local",
            port=0,
            token="test-token",
            auto_run=False,
            llm_profiles_file=Path(self.temp_dir.name) / "profiles.json",
        )
        self.manager = _FakeTaskRuns()
        self.server.service.runtime.task_runs = self.manager
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host, self.port = self.server.server_address

    def teardown_method(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.service.shutdown()
        self.server.server_close()
        self.temp_dir.cleanup()

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> tuple[int, Any]:
        connection = http.client.HTTPConnection(self.host, self.port, timeout=10)
        headers = {"Authorization": "Bearer test-token"}
        payload = None
        if body is not None:
            payload = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        connection.close()
        return response.status, json.loads(raw) if raw else None

    def test_task_run_list_detail_and_ledger_are_paginated_and_redacted(self) -> None:
        self.manager.recovery_option_records = [{
            "schema_version": 1,
            "option_id": "effect_receipt:binding",
            "kind": "effect_receipt",
            "label": "Public receipt label",
            "requires_confirmation": True,
            "requires_receipt": True,
            "receipt_fields": ["receipt_id"],
            "effect_id": "effect-1",
            "expected_transaction_state": "unknown",
            "runtime_epoch": 12,
            "provider_secret": "TASK_RUN_PRIVATE_RECOVERY_SECRET",
            "receipt": {"credential": "TASK_RUN_PRIVATE_RECOVERY_CREDENTIAL"},
        }]
        list_status, page = self.request(
            "GET",
            "/api/task-runs?status=queued,waiting_human&limit=1&cursor=cursor-1",
        )
        detail_status, detail = self.request(
            "GET",
            "/api/task-runs/run-1?requirements_limit=2&requirements_cursor=requirement-cursor",
        )
        ledger_status, ledger = self.request(
            "GET",
            "/api/task-runs/run-1/ledger?limit=1&cursor=ledger-cursor",
        )

        assert (list_status, detail_status, ledger_status) == (200, 200, 200)
        assert page["next_cursor"] == "cursor-2"
        assert page["has_more"] is True
        summary = page["items"][0]
        assert summary["run_id"] == "run-1"
        assert summary["revision"] == 1
        assert summary["status"] == "queued"
        assert summary["allowed_actions"] == ["run", "pause", "cancel"]
        serialized = json.dumps({"page": page, "detail": detail})
        assert "TASK_RUN_PRIVATE" not in serialized
        assert detail["requirements"]["next_cursor"] == "requirement-next"
        assert detail["requirements"]["has_more"] is True
        assert detail["requirements"]["items"][0]["content_text"] == "retained requirement"
        assert detail["requirements"]["items"][0]["content_retention"] == "plaintext"
        assert detail["requirements"]["items"][1]["content_retention"] == "hash_only"
        assert detail["requirements"]["items"][1]["content_available"] is False
        assert "content_text" not in detail["requirements"]["items"][1]
        assert detail["recovery_options"] == [{
            "schema_version": 1,
            "option_id": "effect_receipt:binding",
            "kind": "effect_receipt",
            "label": "Public receipt label",
            "requires_confirmation": True,
            "requires_receipt": True,
            "receipt_fields": ["receipt_id"],
            "effect_id": "effect-1",
            "expected_transaction_state": "unknown",
            "runtime_epoch": 12,
        }]
        assert "TASK_RUN_PRIVATE_RECOVERY" not in serialized
        assert ledger == {
            "items": [
                {
                    "schema_version": 1,
                    "item_id": "ledger-1",
                    "run_id": "run-1",
                    "seq": 1,
                    "kind": "status_transition",
                    "status": "queued",
                    "label": "created",
                    "occurred_at": "2030-01-01T00:00:00+00:00",
                    "metadata": {},
                }
            ],
            "next_cursor": None,
            "has_more": False,
        }
        assert self.manager.calls[0] == (
            "list",
            (),
            {
                "statuses": ("queued", "waiting_human"),
                "limit": 1,
                "cursor": "cursor-1",
            },
        )
        requirements_call = next(
            call for call in self.manager.calls if call[0] == "list_requirements"
        )
        assert requirements_call[2] == {
            "limit": 2,
            "cursor": "requirement-cursor",
        }

    def test_task_run_payload_purge_is_not_an_http_action(self) -> None:
        status, payload = self.request(
            "POST",
            "/api/task-runs/run-1/purge_payloads",
            {"expected_revision": 1, "command_id": "purge-over-http"},
        )

        assert status == 404
        assert payload["error"]["message"] == "unknown task-runs endpoint"

    def test_create_and_mutations_validate_identity_and_confirmation(self) -> None:
        missing_client_status, _ = self.request(
            "POST",
            "/api/task-runs",
            {"spec": {"goal": "work"}},
        )
        created_status, created = self.request(
            "POST",
            "/api/task-runs",
            {
                "spec": {"goal": "work", "display_title": "Work"},
                "client_request_id": "client-1",
                "auto_run": False,
            },
        )
        missing_revision_status, _ = self.request(
            "POST",
            "/api/task-runs/run-1/pause",
            {"command_id": "pause-1"},
        )
        cancel_confirmation_status, cancel_confirmation = self.request(
            "POST",
            "/api/task-runs/run-1/cancel",
            {"expected_revision": 2, "command_id": "cancel-1"},
        )
        recover_confirmation_status, recover_confirmation = self.request(
            "POST",
            "/api/task-runs/run-1/recover",
            {
                "expected_revision": 2,
                "command_id": "recover-1",
                "option_id": "register_receipt",
            },
        )
        pause_status, _ = self.request(
            "POST",
            "/api/task-runs/run-1/pause",
            {"expected_revision": 2, "command_id": "pause-2"},
        )
        resume_status, _ = self.request(
            "POST",
            "/api/task-runs/run-1/resume",
            {"expected_revision": 3, "command_id": "resume-1"},
        )
        confirmed_cancel_status, _ = self.request(
            "POST",
            "/api/task-runs/run-1/cancel",
            {
                "expected_revision": 4,
                "command_id": "cancel-2",
                "confirmed": True,
                "reason": "host requested cancellation",
            },
        )
        rerun_missing_client_status, _ = self.request(
            "POST",
            "/api/task-runs/run-1/rerun",
            {"expected_revision": 5, "command_id": "rerun-1"},
        )

        assert missing_client_status == 400
        assert created_status == 200
        assert created["run_id"] == "run-1"
        assert missing_revision_status == 400
        assert cancel_confirmation_status == 409
        assert cancel_confirmation["error"]["action"] == "task_run.cancel"
        assert recover_confirmation_status == 409
        assert recover_confirmation["error"]["action"] == "task_run.recover"
        assert (pause_status, resume_status, confirmed_cancel_status) == (200, 200, 200)
        assert rerun_missing_client_status == 400
        create = next(call for call in self.manager.calls if call[0] == "create")
        assert create[2] == {"client_request_id": "client-1", "auto_run": False}
        assert next(call for call in self.manager.calls if call[0] == "pause")[2] == {
            "expected_revision": 2,
            "command_id": "pause-2",
        }
        assert next(call for call in self.manager.calls if call[0] == "resume")[2] == {
            "expected_revision": 3,
            "command_id": "resume-1",
        }
        assert next(call for call in self.manager.calls if call[0] == "cancel")[2] == {
            "expected_revision": 4,
            "command_id": "cancel-2",
            "reason": "host requested cancellation",
        }

    @pytest.mark.parametrize(
        ("path", "body", "method_name", "result_run_id"),
        [
            (
                "/api/task-runs",
                {
                    "spec": {"goal": "work", "display_title": "Work"},
                    "client_request_id": "create-latest",
                },
                "create",
                "run-1",
            ),
            (
                "/api/task-runs/run-1/run",
                {"expected_revision": 2, "command_id": "run-latest"},
                "run_until_blocked",
                "run-1",
            ),
            (
                "/api/task-runs/run-1/pause",
                {"expected_revision": 2, "command_id": "pause-latest"},
                "pause",
                "run-1",
            ),
            (
                "/api/task-runs/run-1/resume",
                {"expected_revision": 2, "command_id": "resume-latest"},
                "resume",
                "run-1",
            ),
            (
                "/api/task-runs/run-1/cancel",
                {
                    "expected_revision": 2,
                    "command_id": "cancel-latest",
                    "confirmed": True,
                },
                "cancel",
                "run-1",
            ),
            (
                "/api/task-runs/run-1/follow-ups",
                {
                    "expected_revision": 2,
                    "command_id": "follow-up-latest",
                    "body": "continue",
                },
                "follow_up",
                "run-1",
            ),
            (
                "/api/task-runs/run-1/recover",
                {
                    "expected_revision": 2,
                    "command_id": "recover-latest",
                    "option_id": "linked-rerun",
                    "confirmed": True,
                },
                "recover",
                "run-linked",
            ),
            (
                "/api/task-runs/run-1/rerun",
                {
                    "expected_revision": 2,
                    "command_id": "rerun-latest",
                    "client_request_id": "rerun-latest:create",
                },
                "rerun",
                "run-linked",
            ),
        ],
    )
    def test_successful_mutation_returns_and_publishes_latest_result_run_projection(
        self,
        path: str,
        body: dict[str, Any],
        method_name: str,
        result_run_id: str,
    ) -> None:
        receipt = {**_summary(2), "run_id": result_run_id}
        latest = {**_summary(9, status="paused"), "run_id": result_run_id}
        mutation = getattr(self.manager, method_name)
        get_calls: list[str] = []
        setattr(self.manager, method_name, lambda *args, **kwargs: receipt)

        def get(run_id: str) -> dict[str, Any]:
            get_calls.append(run_id)
            return latest

        self.manager.get = get  # type: ignore[method-assign]

        status, response = self.request("POST", path, body)

        assert status == 200
        assert response["run_id"] == result_run_id
        assert response["revision"] == 9
        assert get_calls == [result_run_id]
        updates = [
            event
            for event in self.server.service.broadcaster.replay_after(0)
            if event.event == "task_run.updated"
        ]
        result_updates = [
            event for event in updates if event.data.get("run_id") == result_run_id
        ]
        assert result_updates
        assert result_updates[-1].data["revision"] == 9
        setattr(self.manager, method_name, mutation)

    def test_create_is_queued_only_and_never_ignores_a_quantum_bound(self) -> None:
        spec = {"goal": "work", "display_title": "Work"}

        auto_status, auto_body = self.request(
            "POST",
            "/api/task-runs",
            {
                "spec": spec,
                "client_request_id": "client-auto",
                "auto_run": True,
            },
        )
        bound_status, bound_body = self.request(
            "POST",
            "/api/task-runs",
            {
                "spec": spec,
                "client_request_id": "client-bound",
                "max_quanta": 3,
            },
        )

        assert (auto_status, bound_status) == (400, 400)
        assert auto_body["error"]["code"] == "task_run_create_auto_run_unsupported"
        assert bound_body["error"]["code"] == "task_run_create_max_quanta_unsupported"
        assert not any(call[0] == "create" for call in self.manager.calls)

    def test_revision_conflict_has_stable_409_envelope(self) -> None:
        self.manager.fail_revision = True

        status, body = self.request(
            "POST",
            "/api/task-runs/run-1/pause",
            {"expected_revision": 8, "command_id": "pause-stale"},
        )

        assert status == 409
        assert body["ok"] is False
        assert body["error"]["code"] == "task_run_revision_conflict"
        assert body["error"]["type"] == "TaskRunRevisionConflict"
        assert body["error"]["command_admitted"] is False

    def test_command_conflict_has_distinct_stable_409_envelope(self) -> None:
        self.manager.fail_command = True

        status, body = self.request(
            "POST",
            "/api/task-runs/run-1/pause",
            {"expected_revision": 8, "command_id": "pause-reused"},
        )

        assert status == 409
        assert body["ok"] is False
        assert body["error"]["code"] == "task_run_command_conflict"
        assert body["error"]["type"] == "TaskRunCommandConflict"

    def test_cancel_can_interrupt_concurrent_run_without_gui_lock_starvation(self) -> None:
        self.manager.run_started = threading.Event()
        self.manager.cancel_releases_run = threading.Event()
        run_response: list[tuple[int, Any]] = []
        run_thread = threading.Thread(
            target=lambda: run_response.append(
                self.request(
                    "POST",
                    "/api/task-runs/run-1/run",
                    {"expected_revision": 1, "command_id": "run-concurrent"},
                )
            ),
            daemon=True,
        )
        run_thread.start()
        assert self.manager.run_started.wait(timeout=2)

        cancel_status, _ = self.request(
            "POST",
            "/api/task-runs/run-1/cancel",
            {
                "expected_revision": 2,
                "command_id": "cancel-concurrent",
                "confirmed": True,
            },
        )

        run_thread.join(timeout=2)
        assert cancel_status == 200
        assert self.manager.cancel_releases_run.is_set()
        assert run_response and run_response[0][0] == 200

    def test_sse_summary_drops_lower_revision_and_never_contains_payload(self) -> None:
        service = self.server.service
        service._publish_task_run_summary(_summary(5, status="running"))
        service._publish_task_run_summary(_summary(4, status="queued"))
        service._publish_task_run_summary(_summary(5, status="running"))

        updates = [
            event
            for event in service.broadcaster.replay_after(0)
            if event.event == "task_run.updated"
        ]
        assert len(updates) == 1
        assert updates[0].data["run_id"] == "run-1"
        assert updates[0].data["revision"] == 5
        assert "TASK_RUN_PRIVATE" not in json.dumps(updates[0].data)

    def test_exact_human_request_get_does_not_depend_on_snapshot_window(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="exact human request reconciliation",
        )
        runtime.capability.grant(
            pid,
            "human:owner",
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        first = runtime.human.ask(pid, "first exact question", blocking=True)
        runtime.human.ask(pid, "second exact question", blocking=True)

        status, body = self.request("GET", f"/api/human-requests/{first}")

        assert status == 200
        assert body["request_id"] == first
        assert body["payload"]["question"] == "first exact question"


def test_real_task_run_manager_round_trips_detail_and_ledger_cursors(
    tmp_path: Path,
) -> None:
    config = replace(
        DEFAULT_CONFIG,
        task_runs=replace(
            DEFAULT_CONFIG.task_runs,
            plaintext_payloads_enabled=True,
        ),
    )
    server = create_gui_http_server(
        db=str(tmp_path / "real-task-run-http.sqlite"),
        port=0,
        token="real-task-run-token",
        auto_run=False,
        config=config,
        llm_profiles_file=tmp_path / "profiles.json",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address

    def request(
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> tuple[int, Any]:
        connection = http.client.HTTPConnection(host, port, timeout=10)
        headers = {"Authorization": "Bearer real-task-run-token"}
        raw = None
        if body is not None:
            raw = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        connection.request(method, path, body=raw, headers=headers)
        response = connection.getresponse()
        payload = response.read()
        connection.close()
        return response.status, json.loads(payload) if payload else None

    try:
        created_status, created = request(
            "POST",
            "/api/task-runs",
            {
                "spec": {
                    "schema_version": 1,
                    "goal": "durable HTTP integration goal",
                    "display_title": "HTTP integration",
                },
                "client_request_id": "http-real-create",
            },
        )
        assert created_status == 200
        run_id = created["run_id"]

        followed_status, followed = request(
            "POST",
            f"/api/task-runs/{run_id}/follow-ups",
            {
                "expected_revision": created["revision"],
                "command_id": "http-real-follow-up",
                "body": "durable follow-up",
                "kind": "normal",
                "required": True,
            },
        )
        assert followed_status == 200
        assert followed["revision"] > created["revision"]

        paused_status, paused = request(
            "POST",
            f"/api/task-runs/{run_id}/pause",
            {
                "expected_revision": followed["revision"],
                "command_id": "http-real-pause",
            },
        )
        assert paused_status == 200
        assert paused["revision"] > followed["revision"]
        replay_status, replayed_follow_up = request(
            "POST",
            f"/api/task-runs/{run_id}/follow-ups",
            {
                "expected_revision": created["revision"],
                "command_id": "http-real-follow-up",
                "body": "durable follow-up",
                "kind": "normal",
                "required": True,
            },
        )
        assert replay_status == 200
        assert replayed_follow_up["revision"] == paused["revision"]
        assert replayed_follow_up["status"] == "paused"
        runtime_receipt = server.service.runtime.task_runs.follow_up(
            run_id,
            body="durable follow-up",
            kind="normal",
            required=True,
            expected_revision=created["revision"],
            command_id="http-real-follow-up",
        )
        assert runtime_receipt.revision == followed["revision"]
        resumed_status, resumed = request(
            "POST",
            f"/api/task-runs/{run_id}/resume",
            {
                "expected_revision": paused["revision"],
                "command_id": "http-real-resume",
            },
        )
        assert resumed_status == 200
        assert resumed["revision"] > paused["revision"]

        detail_status, first_detail = request(
            "GET",
            f"/api/task-runs/{run_id}?requirements_limit=1",
        )
        assert detail_status == 200
        assert first_detail["requirements"]["has_more"] is True
        assert first_detail["requirements"]["items"][0]["content_retention"] == "plaintext"
        requirement_cursor = first_detail["requirements"]["next_cursor"]
        next_detail_status, next_detail = request(
            "GET",
            f"/api/task-runs/{run_id}?requirements_limit=1&requirements_cursor={requirement_cursor}",
        )
        assert next_detail_status == 200
        assert next_detail["requirements"]["items"][0]["kind"] == "follow_up"

        ledger_status, first_ledger = request(
            "GET",
            f"/api/task-runs/{run_id}/ledger?limit=1",
        )
        assert ledger_status == 200
        assert first_ledger["has_more"] is True
        ledger_cursor = first_ledger["next_cursor"]
        next_ledger_status, next_ledger = request(
            "GET",
            f"/api/task-runs/{run_id}/ledger?limit=1&cursor={ledger_cursor}",
        )
        assert next_ledger_status == 200
        assert next_ledger["items"][0]["seq"] > first_ledger["items"][0]["seq"]

        server.service.runtime.capability.grant(
            created["root_pid"],
            "human:owner",
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        request_id = server.service.runtime.human.ask(
            created["root_pid"],
            "durable HTTP human question",
            blocking=True,
        )
        human_status, human_page = request(
            "GET",
            f"/api/task-runs/{run_id}/human-requests?limit=1",
        )
        assert human_status == 200
        assert human_page["presentation_truncated"] is False
        assert human_page["items"][0]["request_id"] == request_id
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.service.shutdown()
        server.server_close()
