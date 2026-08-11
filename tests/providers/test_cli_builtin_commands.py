from __future__ import annotations
import asyncio
import pytest
import contextlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from agent_libos import Runtime
from agent_libos.api.cli import cli as cli_entrypoint
from agent_libos.api.cli import _handle_interactive_human_response
from agent_libos.api.cli import _handle_interactive_line
from agent_libos.api.cli import main as cli_main
from agent_libos.api.cli import _print_interactive_help
from agent_libos.api.cli import _parse_cli_args
from agent_libos.api.cli import _run_capabilities_command
from agent_libos.api.cli import _run_interactive_command
from agent_libos.api.cli import _run_mcp_command
from agent_libos.api.cli import _show_pending_interactive_human_request
from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import AgentLibOSConfigDeprecationWarning, DEFAULT_CONFIG
from agent_libos.models import (
    CapabilityRight,
    ObjectMetadata,
    ObjectType,
    HumanRequestStatus,
    McpConnectionInfo,
    McpDiscoveryResult,
    McpExchangePhase,
    McpExchangeReceipt,
    McpProtocolEra,
    McpProtocolMode,
    ProcessMessageKind,
    ProcessStatus,
    process_outcome_to_mapping,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.substrate import LocalResourceProviderSubstrate
from tests.support.mcp import MCP_TEST_STDIO_COMMAND, MCP_TEST_STDIO_COMMAND_YAML


def _create_store_with_schema_version(db: Path, version: int) -> None:
    connection = sqlite3.connect(db)
    try:
        connection.execute(
            "CREATE TABLE runtime_schema ("
            "singleton INTEGER PRIMARY KEY, "
            "schema_version INTEGER NOT NULL)"
        )
        connection.execute(
            "INSERT INTO runtime_schema (singleton, schema_version) VALUES (1, ?)",
            (version,),
        )
        connection.commit()
    finally:
        connection.close()


class TestCLIBuiltinCommand:

    @pytest.mark.parametrize("requested, expected", [(None, 100), (7, 7)])
    def test_cli_capability_list_validates_and_pushes_limit_to_store(
        self,
        monkeypatch: pytest.MonkeyPatch,
        requested: int | None,
        expected: int,
    ) -> None:
        runtime = Runtime.open("local")
        observed: list[tuple[str | None, int | None]] = []

        def record_list(
            subject: str | None = None,
            *,
            limit: int | None = None,
        ) -> list[object]:
            observed.append((subject, limit))
            return []

        monkeypatch.setattr(runtime.store, "list_capabilities", record_list)
        try:
            result = _run_capabilities_command(
                runtime,
                SimpleNamespace(
                    actor_pid=None,
                    capabilities_command="list",
                    subject=None,
                    include_inactive=True,
                    limit=requested,
                ),
            )
            assert result == []
            assert observed == [(None, expected)]
        finally:
            runtime.close()

    @pytest.mark.parametrize("limit", [True, 0, -1, 101, "1", 1.0])
    def test_cli_capability_list_rejects_invalid_or_oversized_limit(
        self,
        limit: object,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            with pytest.raises(ValidationError, match="capability list limit"):
                _run_capabilities_command(
                    runtime,
                    SimpleNamespace(
                        actor_pid=None,
                        capabilities_command="list",
                        subject=None,
                        include_inactive=True,
                        limit=limit,
                    ),
                )
        finally:
            runtime.close()

    def test_interactive_run_uses_configured_default_quantum_budget_when_omitted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = replace(
            DEFAULT_CONFIG,
            runtime=replace(DEFAULT_CONFIG.runtime, run_until_idle_max_quanta=7),
        )
        runtime = Runtime.open("local", config=config)
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="interactive budget")
            observed: list[int | None] = []

            async def record_scheduler_call(
                _runner: object,
                *,
                max_quanta: int | None = None,
            ) -> list[object]:
                observed.append(max_quanta)
                return []

            commands = iter((None, "exit"))
            monkeypatch.setattr(runtime.scheduler, "arun_until_idle", record_scheduler_call)
            monkeypatch.setattr("agent_libos.api.cli._redirect_human_output_to_stderr", lambda _runtime: None)
            monkeypatch.setattr("agent_libos.api.cli._start_interactive_input_thread", lambda *_args: None)
            monkeypatch.setattr("agent_libos.api.cli._print_interactive_help", lambda _pid: None)
            monkeypatch.setattr("agent_libos.api.cli._drain_interactive_queue", lambda *_args: next(commands))

            report = asyncio.run(
                _run_interactive_command(
                    runtime,
                    SimpleNamespace(
                        pid=pid,
                        max_quanta=None,
                        human=None,
                        message_channel="human",
                    ),
                )
            )

            assert observed == [7]
            assert report["remaining_quanta"] == 7
            assert report["results"] == []
        finally:
            runtime.close()

    def test_interactive_help_lists_process_and_pending_request_commands(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _print_interactive_help("pid-help")

        output = capsys.readouterr()
        assert output.out == ""
        for command in (
            "/message <text>",
            "/interrupt <text>",
            "/pid <pid>",
            "/answer <text>",
            "/approve",
            "/reject",
            "/allow",
            "/ask",
            "/help",
            "/exit",
        ):
            assert command in output.err

    def test_interactive_external_approval_retains_display_fence_and_refreshes_after_cas_conflict(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        request = SimpleNamespace(
            request_id="req-external",
            human="operator",
            status=HumanRequestStatus.PENDING,
            revision=7,
            payload={"type": "external_operation_approval"},
        )

        class FakeHuman:
            def __init__(self) -> None:
                self.approval_fences: list[dict[str, object]] = []

            def pending(self, *, human: str) -> list[object]:
                assert human == "operator"
                return [request]

            def get(self, request_id: str) -> object:
                assert request_id == request.request_id
                return request

            def present_terminal_request(
                self,
                selected: object,
                *,
                suffix: str,
            ) -> dict[str, object]:
                assert selected is request
                assert "approve" in suffix
                digest = "a" * 64 if request.revision == 7 else "b" * 64
                return {
                    "expected_revision": request.revision,
                    "preview_sha256": digest,
                }

            def approve(
                self,
                request_id: str,
                _decision: dict[str, object],
                *,
                responder: str,
                **fence: object,
            ) -> None:
                assert request_id == request.request_id
                assert responder == "human:operator"
                self.approval_fences.append(fence)
                raise ValidationError(
                    "human request revision conflict: expected 7, found 8"
                )

        fake_human = FakeHuman()
        runtime = SimpleNamespace(human=fake_human)
        state: dict[str, object] = {
            "pid": "pid-external",
            "shown_request_id": "",
            "shown_request_revision": None,
            "shown_preview_sha256": "",
        }

        _show_pending_interactive_human_request(runtime, "operator", state)
        assert state["shown_request_revision"] == 7
        assert state["shown_preview_sha256"] == "a" * 64

        request.revision = 8
        assert (
            _handle_interactive_line(
                runtime,
                "/approve",
                state,
                "operator",
                "human",
                [],
            )
            is None
        )
        assert fake_human.approval_fences == [
            {
                "expected_revision": 7,
                "preview_sha256": "a" * 64,
            }
        ]
        assert state["shown_request_id"] == ""

        _show_pending_interactive_human_request(runtime, "operator", state)
        assert state["shown_request_revision"] == 8
        assert state["shown_preview_sha256"] == "b" * 64
        assert "review the refreshed request" in capsys.readouterr().err

    def test_interactive_approve_reports_atomic_host_policy_rejection(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        request = SimpleNamespace(
            request_id="req-hard-deny",
            human="operator",
            status=HumanRequestStatus.PENDING,
            payload={"type": "external_operation_approval"},
        )

        class FakeHuman:
            def get(self, request_id: str) -> object:
                assert request_id == request.request_id
                return request

            def approve(
                self,
                request_id: str,
                _decision: dict[str, object],
                *,
                responder: str,
                expected_revision: int | None,
                preview_sha256: str | None,
            ) -> object:
                assert request_id == request.request_id
                assert responder == "human:operator"
                assert expected_revision == 3
                assert preview_sha256 == "c" * 64
                return SimpleNamespace(
                    status=HumanRequestStatus.REJECTED,
                    decision={"approved": False, "source": "machine_policy"},
                )

        handled = _handle_interactive_human_response(
            SimpleNamespace(human=FakeHuman()),
            "/approve",
            "operator",
            shown_request_id=request.request_id,
            shown_request_revision=3,
            shown_preview_sha256="c" * 64,
        )

        assert handled is True
        output = capsys.readouterr().err
        assert "Host policy rejected human request req-hard-deny" in output
        assert "Approved human request" not in output

    @pytest.mark.parametrize(
        ("request_type", "response", "message"),
        (
            ("approval", "/allow", "Approval response must be /approve or /reject"),
            ("permission_request", "/answer maybe", "Permission response must be /allow, /ask, or /reject"),
        ),
        ids=("ordinary-approval", "permission"),
    )
    def test_invalid_interactive_decision_keeps_request_pending(
        self,
        request_type: str,
        response: str,
        message: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        request = SimpleNamespace(
            request_id="req-pending",
            human="operator",
            status=HumanRequestStatus.PENDING,
            payload={"type": request_type},
        )

        class FakeHuman:
            def get(self, request_id: str) -> object:
                assert request_id == request.request_id
                return request

            def approve(self, *args: object, **kwargs: object) -> None:
                raise AssertionError("invalid input must not approve the request")

            def reject(self, *args: object, **kwargs: object) -> None:
                raise AssertionError("invalid input must not reject the request")

        handled = _handle_interactive_human_response(
            SimpleNamespace(human=FakeHuman()),
            response,
            "operator",
            shown_request_id=request.request_id,
        )

        assert handled is True
        assert message in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("response", "expected_status", "approved"),
        (
            ("/approve", HumanRequestStatus.APPROVED, True),
            ("/reject", HumanRequestStatus.REJECTED, False),
        ),
        ids=("approve", "reject"),
    )
    def test_interactive_ordinary_decision_persists_request_status(
        self,
        response: str,
        expected_status: HumanRequestStatus,
        approved: bool,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="interactive approval")
            request_id = runtime.human.query(
                pid,
                "owner",
                {"type": "approval", "question": "Proceed with the reviewed action?"},
                blocking=False,
            )

            handled = _handle_interactive_human_response(
                runtime,
                response,
                "owner",
                shown_request_id=request_id,
            )

            persisted = runtime.human.get(request_id)
            assert handled is True
            assert persisted.status is expected_status
            assert persisted.decision == {
                "approved": approved,
                "source": "interactive_cli",
            }
            assert request_id not in {
                request.request_id for request in runtime.human.pending()
            }
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        ("response", "policy", "expected_status", "approved"),
        (
            (
                "/allow",
                CapabilityManager.ALWAYS_ALLOW,
                HumanRequestStatus.APPROVED,
                True,
            ),
            (
                "/ask",
                CapabilityManager.ASK_EACH_TIME,
                HumanRequestStatus.APPROVED,
                True,
            ),
            (
                "/reject",
                CapabilityManager.ALWAYS_DENY,
                HumanRequestStatus.REJECTED,
                False,
            ),
        ),
        ids=("allow", "ask", "reject"),
    )
    def test_interactive_permission_decision_persists_status_and_policy(
        self,
        response: str,
        policy: str,
        expected_status: HumanRequestStatus,
        approved: bool,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(
                image="review-agent:v0",
                goal="interactive permission",
                authority_manifest={
                    "authorized_capabilities": [
                        {
                            "resource": "human:owner",
                            "rights": [CapabilityRight.WRITE.value],
                        }
                    ],
                    "approval_policy": {
                        "requestable_capabilities": [
                            {
                                "resource": "filesystem:workspace:*",
                                "rights": [CapabilityRight.WRITE.value],
                            }
                        ]
                    },
                },
            )
            runtime.capability.grant(
                pid,
                "human:owner",
                [CapabilityRight.WRITE],
                issued_by="test",
            )
            resource = runtime.filesystem.resource_for(
                "agent_outputs/interactive_permission.txt"
            )
            request_id = runtime.human.request_permission(
                pid,
                "owner",
                resource,
                [CapabilityRight.WRITE.value],
                "edit the reviewed output",
                blocking=False,
            )

            handled = _handle_interactive_human_response(
                runtime,
                response,
                "owner",
                shown_request_id=request_id,
            )

            persisted = runtime.human.get(request_id)
            assert handled is True
            assert persisted.status is expected_status
            assert persisted.decision == {
                "approved": approved,
                "policy": policy,
                "source": "interactive_cli",
            }
            assert (
                runtime.capability.permission_policy(
                    pid,
                    resource,
                    CapabilityRight.WRITE,
                )
                == policy
            )
            assert request_id not in {
                request.request_id for request in runtime.human.pending()
            }
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        ("group", "message"),
        (
            (
                "jsonrpc",
                "jsonrpc call --actor-pid must match",
            ),
            (
                "mcp",
                "mcp call --actor-pid must match",
            ),
        ),
    )
    def test_remote_call_actor_pid_cannot_masquerade_as_the_target_process(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        group: str,
        message: str,
    ) -> None:
        db = tmp_path / f"{group}.sqlite"
        runtime = Runtime.open(str(db))

        class NoDispatchProvider:
            def __init__(self) -> None:
                self.interactions: list[str] = []

            def __getattr__(self, name: str):
                def fail_if_called(*_args: object, **_kwargs: object) -> None:
                    self.interactions.append(name)
                    raise AssertionError(f"provider interaction must not occur: {name}")

                return fail_if_called

        try:
            target_pid = runtime.process.spawn(image='base-agent:v0', goal=f'{group} target')
            actor_pid = runtime.process.spawn(image='base-agent:v0', goal=f'{group} mismatched actor')
            remote_id = f'cli-{group}-actor-mismatch'
            provider = NoDispatchProvider()
            protected_caps = []
            if group == 'jsonrpc':
                runtime.jsonrpc.register_endpoint_from_yaml_text(
                    _cli_jsonrpc_manifest(remote_id),
                    actor='test',
                    require_capability=False,
                )
                protected_caps.append(
                    runtime.capability.grant_once(
                        target_pid,
                        f'jsonrpc:{remote_id}:echo',
                        [CapabilityRight.READ],
                        issued_by='test',
                    )
                )
                runtime.jsonrpc.provider = provider
            else:
                runtime.mcp.register_server_from_yaml_text(
                    _cli_mcp_manifest(remote_id),
                    actor='test',
                    require_capability=False,
                )
                protected_caps.extend(
                    [
                        runtime.capability.grant_once(
                            target_pid,
                            f'mcp:{remote_id}:echo',
                            [CapabilityRight.READ],
                            issued_by='test',
                        ),
                        runtime.capability.grant_once(
                            target_pid,
                            'process:spawn',
                            [CapabilityRight.WRITE],
                            issued_by='test',
                        ),
                        runtime.capability.grant_once(
                            target_pid,
                            runtime.mcp.stdio_resource_for_argv(MCP_TEST_STDIO_COMMAND, ['-m', 'demo_mcp']),
                            [CapabilityRight.EXECUTE],
                            issued_by='test',
                        ),
                    ]
                )
                runtime.mcp.provider = provider

            before_effects = runtime.store.list_external_effects(pid=target_pid)
            before_capabilities = {
                cap.cap_id: runtime.store.get_capability(cap.cap_id)
                for cap in protected_caps
            }
            before_audit = runtime.audit.trace()
            before_events = runtime.events.list()
            monkeypatch.setattr('agent_libos.api.cli.Runtime.open', lambda *args, **kwargs: runtime)
            monkeypatch.setattr(runtime, 'shutdown', lambda **_kwargs: {'ok': True})

            with pytest.raises(SystemExit, match=message):
                cli_main(
                    [
                        "--db",
                        str(db),
                        group,
                        "--actor-pid",
                        actor_pid,
                        "call",
                        target_pid,
                        remote_id,
                        "echo",
                    ]
                )

            assert provider.interactions == []
            assert runtime.store.list_external_effects(pid=target_pid) == before_effects
            assert {
                cap_id: runtime.store.get_capability(cap_id)
                for cap_id in before_capabilities
            } == before_capabilities
            assert all(capability is not None and capability.uses_remaining == 1 for capability in before_capabilities.values())
            assert runtime.audit.trace() == before_audit
            assert runtime.events.list() == before_events
        finally:
            runtime.close()

    def test_cli_processes_preserves_typed_wait_and_outcome_discriminators(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")
        waiting_pid = runtime.process.spawn(goal="typed waiting state")
        runtime.process.pause(waiting_pid, "review")
        terminal_pid = runtime.process.spawn(goal="typed terminal outcome")
        runtime.process.exit(terminal_pid, message="done")
        monkeypatch.setattr("agent_libos.api.cli.Runtime.open", lambda *args, **kwargs: runtime)
        stdout = io.StringIO()
        stderr = io.StringIO()

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            cli_entrypoint(["processes"])

        processes = {
            process["pid"]: process
            for process in json.loads(stdout.getvalue())
        }
        assert stderr.getvalue() == ""
        assert processes[waiting_pid]["wait_state"] == {
            "schema_version": 1,
            "kind": "paused",
            "reason_oid": processes[waiting_pid]["wait_state"]["reason_oid"],
        }
        assert processes[waiting_pid]["outcome"] is None
        assert processes[terminal_pid]["wait_state"] is None
        assert processes[terminal_pid]["outcome"] == {
            "schema_version": 1,
            "kind": "exited",
            "result_oid": processes[terminal_pid]["outcome"]["result_oid"],
        }
        assert processes[terminal_pid]["outcome"]["result_oid"]

    @pytest.mark.parametrize(
        ("arguments", "message"),
        (
            (["--limit", "0"], "payload retention request limit must be positive"),
            (
                ["--after-created-at", "", "--after-record-id", "record-1"],
                "payload retention cursor created_at must not be empty",
            ),
            (["--limit", "3"], "payload retention request limit exceeds hard limit 2"),
        ),
    )
    def test_cli_payload_retention_value_errors_use_structured_error_boundary(
        self,
        monkeypatch: pytest.MonkeyPatch,
        arguments: list[str],
        message: str,
    ) -> None:
        config = replace(
            DEFAULT_CONFIG,
            runtime=replace(
                DEFAULT_CONFIG.runtime,
                payload_retention_enabled=True,
                payload_retention_summary_after_seconds=0,
                payload_retention_page_size=1,
                payload_retention_page_hard_limit=2,
            ),
        )
        runtime = Runtime.open("local", config=config)
        monkeypatch.setattr("agent_libos.api.cli.Runtime.open", lambda *args, **kwargs: runtime)
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
            pytest.raises(SystemExit) as raised,
        ):
            cli_entrypoint(["payload-retention", "llm_call", *arguments])

        assert raised.value.code == 1
        assert stderr.getvalue() == ""
        assert json.loads(stdout.getvalue()) == {
            "schema_version": 1,
            "error": {
                "type": "ValidationError",
                "message": message,
            },
        }

    def test_cli_runtime_not_found_uses_structured_error_and_exit_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")
        monkeypatch.setattr("agent_libos.api.cli.Runtime.open", lambda *args, **kwargs: runtime)
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout), pytest.raises(SystemExit) as raised:
            cli_entrypoint(["resources", "missing-pid"])

        assert raised.value.code == 1
        error = json.loads(stdout.getvalue())["error"]
        assert error == {
            "type": "NotFound",
            "message": "process not found: missing-pid",
        }

    def test_cli_missing_capability_uses_structured_error_and_exit_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")
        monkeypatch.setattr("agent_libos.api.cli.Runtime.open", lambda *args, **kwargs: runtime)
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
            pytest.raises(SystemExit) as raised,
        ):
            cli_entrypoint(["capabilities", "inspect", "cap-missing"])

        assert raised.value.code == 1
        assert stderr.getvalue() == ""
        assert json.loads(stdout.getvalue()) == {
            "schema_version": 1,
            "error": {
                "type": "NotFound",
                "message": "capability not found: cap-missing",
            },
        }

    def test_cli_unsupported_store_version_uses_structured_error_and_exit_one(
        self,
    ) -> None:
        message = (
            "Agent libOS store schema v3 is not writable or readable by this runtime; "
            "expected 7. Use Agent libOS 1.0.1 to view or archive this store. "
            "No migration was attempted."
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Path(temp_dir) / "unsupported.sqlite"
            _create_store_with_schema_version(db, 3)
            stdout = io.StringIO()
            stderr = io.StringIO()

            with (
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
                pytest.raises(SystemExit) as raised,
            ):
                cli_entrypoint(["--db", str(db), "init"])

            assert raised.value.code == 1
            assert stderr.getvalue() == ""
            error = json.loads(stdout.getvalue())["error"]
            assert error == {
                "type": "UnsupportedStoreVersion",
                "message": message,
            }

    def test_python_module_entrypoint_uses_structured_error_boundary(self) -> None:
        message = (
            "Agent libOS store schema v3 is not writable or readable by this runtime; "
            "expected 7. Use Agent libOS 1.0.1 to view or archive this store. "
            "No migration was attempted."
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Path(temp_dir) / "unsupported.sqlite"
            _create_store_with_schema_version(db, 3)

            result = subprocess.run(
                [sys.executable, "-m", "agent_libos", "--db", str(db), "init"],
                capture_output=True,
                text=True,
                check=False,
            )

        assert result.returncode == 1
        assert result.stderr == ""
        error = json.loads(result.stdout)["error"]
        assert error == {
            "type": "UnsupportedStoreVersion",
            "message": message,
        }

    def test_cli_ignores_config_yaml_from_cwd_for_default_image(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = root / 'runtime.sqlite'
            root.joinpath('config.yaml').write_text(
                'runtime:\n  default_image_id: configured-base:v0\n',
                encoding='utf-8',
            )
            monkeypatch.setattr('agent_libos.api.cli.load_config_from_project_root', lambda: DEFAULT_CONFIG)

            with _temporary_cwd(root):
                result = _run_cli_json(['--db', str(db), 'spawn', '--goal', 'configured default'])

            assert result['image'] == DEFAULT_CONFIG.runtime.default_image_id

    def test_cli_config_argument_overrides_default_project_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = root / 'runtime.sqlite'
            root.joinpath('config.yaml').write_text(
                'runtime:\n  default_image_id: cwd-base:v0\n',
                encoding='utf-8',
            )
            alt = root / 'alt-config.yaml'
            alt.write_text(
                'runtime:\n  default_image_id: alt-base:v0\n',
                encoding='utf-8',
            )

            with _temporary_cwd(root):
                result = _run_cli_json(['--config', str(alt), '--db', str(db), 'spawn', '--goal', 'configured default'])

            assert result['image'] == 'alt-base:v0'

    def test_cli_explicit_db_overrides_configured_local_store_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            configured = root / 'configured.sqlite'
            explicit = root / 'explicit.sqlite'
            config = root / 'config.yaml'
            config.write_text(
                f'runtime:\n  local_store_target: {configured.as_posix()}\n',
                encoding='utf-8',
            )

            with _temporary_cwd(root):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    cli_main(['--config', str(config), '--db', str(explicit), 'init'])

            assert stdout.getvalue().strip() == f'initialized {explicit}'
            assert explicit.exists()
            assert not configured.exists()

    def test_cli_uses_configured_local_store_target_when_db_is_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            configured = root / 'configured.sqlite'
            config = root / 'config.yaml'
            config.write_text(
                f'runtime:\n  local_store_target: {configured.as_posix()}\n',
                encoding='utf-8',
            )

            with _temporary_cwd(root):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    cli_main(['--config', str(config), 'init'])

            assert stdout.getvalue().strip() == f'initialized {configured.as_posix()}'
            assert configured.exists()

    def test_cli_uses_configured_postgres_dsn_when_db_is_omitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / 'config.yaml'
            config.write_text(
                'runtime:\n'
                '  store_backend: postgres\n'
                '  store_dsn: "postgresql://agent:secret@localhost/agent_libos"\n',
                encoding='utf-8',
            )
            calls: dict[str, object] = {}

            class DummyRuntime:
                def shutdown(self, *, actor: str, reason: str) -> dict[str, bool]:
                    calls['shutdown'] = (actor, reason)
                    return {'ok': True}

            def fake_open(target: object = None, **kwargs: object) -> DummyRuntime:
                calls['target'] = target
                calls['config'] = kwargs.get('config')
                return DummyRuntime()

            monkeypatch.setattr(Runtime, 'open', staticmethod(fake_open))

            with _temporary_cwd(root):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    cli_main(['--config', str(config), 'init'])

            assert calls['target'] is None
            assert stdout.getvalue().strip() == 'initialized postgresql://***@localhost/agent_libos'

    def test_cli_rejects_postgres_backend_without_store_dsn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / 'config.yaml'
            config.write_text('runtime:\n  store_backend: postgres\n', encoding='utf-8')

            with _temporary_cwd(root):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), pytest.raises(SystemExit) as raised:
                    cli_main(['--config', str(config), 'init'])

            assert raised.value.code == 2
            assert 'runtime.store_dsn is required' in stderr.getvalue()

    def test_cli_passes_explicit_postgres_db_to_runtime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dsn = 'postgresql://agent:secret@localhost/agent_libos'
        calls: dict[str, object] = {}

        class DummyRuntime:
            def shutdown(self, *, actor: str, reason: str) -> dict[str, bool]:
                calls['shutdown'] = (actor, reason)
                return {'ok': True}

        def fake_open(target: object = None, **kwargs: object) -> DummyRuntime:
            calls['target'] = target
            calls['config'] = kwargs.get('config')
            return DummyRuntime()

        monkeypatch.setattr(Runtime, 'open', staticmethod(fake_open))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli_main(['--db', dsn, 'init'])

        assert calls['target'] == dsn
        assert stdout.getvalue().strip() == 'initialized postgresql://***@localhost/agent_libos'

    def test_cli_fails_visibly_when_runtime_reports_incomplete_shutdown(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class DummyRuntime:
            def shutdown(self, *, actor: str, reason: str) -> dict[str, object]:
                return {'ok': False, 'errors': ['injected teardown failure']}

        monkeypatch.setattr(Runtime, 'open', staticmethod(lambda *_args, **_kwargs: DummyRuntime()))

        with pytest.raises(RuntimeError, match='teardown remained incomplete'):
            cli_main(['init'])

    def test_cli_propagates_runtime_shutdown_exception_after_successful_command(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        shutdown_error = OSError('injected shutdown exception')

        class DummyRuntime:
            def shutdown(self, *, actor: str, reason: str) -> dict[str, object]:
                raise shutdown_error

        monkeypatch.setattr(Runtime, 'open', staticmethod(lambda *_args, **_kwargs: DummyRuntime()))

        with pytest.raises(OSError) as raised:
            cli_main(['init'])

        assert raised.value is shutdown_error

    def test_cli_preserves_primary_command_error_when_shutdown_also_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        primary_error = RuntimeError('injected command failure')

        class DummyRuntime:
            def shutdown(self, *, actor: str, reason: str) -> dict[str, object]:
                return {'ok': False, 'errors': ['secondary teardown failure']}

        def fail_demo(_runtime: object) -> dict[str, object]:
            raise primary_error

        monkeypatch.setattr(Runtime, 'open', staticmethod(lambda *_args, **_kwargs: DummyRuntime()))
        monkeypatch.setattr('agent_libos.api.cli.run_demo', fail_demo)

        with pytest.raises(RuntimeError) as raised:
            cli_main(['demo'])

        assert raised.value is primary_error
        assert any('runtime teardown also failed' in note for note in primary_error.__notes__)

    def test_cli_cd_changes_process_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / 'pkg').mkdir()
            db = root / 'runtime.sqlite'
            with _temporary_cwd(root):
                spawn = _run_cli_json(['--db', str(db), 'spawn', '--image', 'review-agent:v0', '--goal', 'set cwd'])
                _run_cli_json([
                    '--db',
                    str(db),
                    'capabilities',
                    'grant',
                    spawn['pid'],
                    'filesystem:workspace:pkg/*',
                    '--rights',
                    'read',
                ])
                result = _run_cli_json(['--db', str(db), 'cd', spawn['pid'], 'pkg'])
            runtime = Runtime.open(db, substrate=LocalResourceProviderSubstrate(root))
            try:
                assert result['pid'] == spawn['pid']
                assert result['working_directory'] == 'pkg'
                assert runtime.process.get(spawn['pid']).working_directory == 'pkg'
            finally:
                runtime.close()

    def test_cli_spawn_and_exec_accept_llm_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = root / 'runtime.sqlite'
            with _temporary_cwd(root):
                spawn = _run_cli_json([
                    '--db',
                    str(db),
                    'spawn',
                    '--image',
                    'base-agent:v0',
                    '--goal',
                    'profiled',
                    '--llm-profile',
                    'cli-spawn',
                ])
                result = _run_cli_json([
                    '--db',
                    str(db),
                    'exec',
                    'base-agent:v0',
                    'profiled exec',
                    '--pid',
                    spawn['pid'],
                    '--llm-profile',
                    'cli-exec',
                    '--no-run',
                ])
            runtime = Runtime.open(db, substrate=LocalResourceProviderSubstrate(root))
            try:
                process = runtime.process.get(spawn['pid'])
                assert spawn['llm_profile_id'] == 'cli-spawn'
                assert result['exec']['old_llm_profile_id'] == 'cli-spawn'
                assert result['exec']['new_llm_profile_id'] == 'cli-exec'
                assert process.llm_profile_id == 'cli-exec'
            finally:
                runtime.close()

    def test_cli_exit_marks_process_exited_with_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = root / 'runtime.sqlite'
            with _temporary_cwd(root):
                spawn = _run_cli_json(['--db', str(db), 'spawn', '--image', 'base-agent:v0', '--goal', 'finish'])
                result = _run_cli_json(['--db', str(db), 'exit', spawn['pid'], '--payload', '{"done": true}'])
            runtime = Runtime.open(db, substrate=LocalResourceProviderSubstrate(root))
            try:
                process = runtime.process.get(spawn['pid'])
                assert result['pid'] == spawn['pid']
                assert result['status'] == ProcessStatus.EXITED.value
                assert result['result_oid'] is not None
                assert process.status == ProcessStatus.EXITED
                assert (process.status_message or '').startswith('result_oid:')
            finally:
                runtime.close()

    def test_cli_exit_message_returns_the_durable_typed_outcome(self) -> None:
        for failed in (False, True):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                db = root / 'runtime.sqlite'
                with _temporary_cwd(root):
                    spawn = _run_cli_json(
                        [
                            '--db',
                            str(db),
                            'spawn',
                            '--image',
                            'base-agent:v0',
                            '--goal',
                            'finish with a message',
                        ]
                    )
                    arguments = [
                        '--db',
                        str(db),
                        'exit',
                        spawn['pid'],
                        '--message',
                        'done',
                    ]
                    if failed:
                        arguments.append('--failed')
                    result = _run_cli_json(arguments)
                runtime = Runtime.open(
                    db,
                    substrate=LocalResourceProviderSubstrate(root),
                )
                try:
                    process = runtime.process.get(spawn['pid'])
                    expected_outcome = process_outcome_to_mapping(process.outcome)
                    assert result['status'] == (
                        ProcessStatus.FAILED.value
                        if failed
                        else ProcessStatus.EXITED.value
                    )
                    assert result['wait_state'] is None
                    assert result['outcome'] == expected_outcome
                    assert result['state_generation'] == process.state_generation
                    assert result['result_oid'] == expected_outcome['result_oid']
                    assert result['result_oid'] is not None
                finally:
                    runtime.close()

    def test_cli_exec_loads_image_package_from_first_arg_and_uses_second_arg_as_goal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = root / 'runtime.sqlite'
            package = root / 'cli-image'
            _write_cli_image_package(package)
            with _temporary_cwd(root):
                spawn = _run_cli_json(['--db', str(db), 'spawn', '--image', 'base-agent:v0', '--goal', 'old goal'])
                before = Runtime.open(db, substrate=LocalResourceProviderSubstrate(root))
                try:
                    old_goal_oid = before.process.get(spawn['pid']).goal_oid
                finally:
                    before.close()
                result = _run_cli_json(['--db', str(db), 'exec', str(package), 'new goal from first arg', '--pid', spawn['pid'], '--no-run'])
            runtime = Runtime.open(db, substrate=LocalResourceProviderSubstrate(root))
            try:
                process = runtime.process.get(spawn['pid'])
                assert result['goal'] == 'new goal from first arg'
                assert result['image_arg'] == str(package)
                assert result['loaded_image']['image_id'] == 'cli-package-agent:v0'
                assert result['process']['image'] == 'cli-package-agent:v0'
                assert not result['ran']
                assert process.image_id == 'cli-package-agent:v0'
                assert process.goal_oid != old_goal_oid
                assert 'human_output' in process.tool_table
            finally:
                runtime.close()

    def test_cli_message_and_interrupt_post_human_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = root / 'runtime.sqlite'
            with _temporary_cwd(root):
                spawn = _run_cli_json(['--db', str(db), 'spawn', '--image', 'base-agent:v0', '--goal', 'listen'])
                normal = _run_cli_json(['--db', str(db), 'message', spawn['pid'], 'please inspect the latest result', '--subject', 'status'])
                interrupt = _run_cli_json(['--db', str(db), 'interrupt', spawn['pid'], 'stop and read this first'])
            runtime = Runtime.open(db, substrate=LocalResourceProviderSubstrate(root))
            try:
                unread = runtime.messages.unread(spawn['pid'])
                assert normal['message']['kind'] == ProcessMessageKind.NORMAL.value
                assert interrupt['message']['kind'] == ProcessMessageKind.INTERRUPT.value
                assert [message.message_id for message in unread] == [normal['message']['message_id'], interrupt['message']['message_id']]
                assert unread[0].sender == 'human:owner'
                assert unread[0].subject == 'status'
                assert unread[1].subject == 'Human interrupt'
            finally:
                runtime.close()

    def test_cli_workflow_run_prints_result_and_persists_exited_process(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = root / 'runtime.sqlite'
            with _temporary_cwd(root):
                result = _run_cli_json(['--db', str(db), 'workflow', 'run', 'get_working_directory'])
            runtime = Runtime.open(db, substrate=LocalResourceProviderSubstrate(root))
            try:
                assert result['ok'] is True
                assert result['tool'] == 'get_working_directory'
                assert result['status'] == ProcessStatus.EXITED.value
                assert result['result_oid'] is not None
                assert runtime.process.get(str(result['pid'])).status == ProcessStatus.EXITED
            finally:
                runtime.close()

    def test_cli_workflow_run_failure_exits_nonzero_after_printing_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = root / 'runtime.sqlite'
            with _temporary_cwd(root):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), pytest.raises(SystemExit) as raised:
                    cli_main([
                        '--db',
                        str(db),
                        'workflow',
                        'run',
                        'parse_pytest_log',
                        '--args-json',
                        '{"log": "FAILED tests/x.py::test_y"}',
                    ])
            assert raised.value.code == 1
            result = json.loads(stdout.getvalue())
            assert result['ok'] is False
            assert result['status'] == ProcessStatus.FAILED.value
            assert 'not in process tool table' in result['error']

    def test_cli_object_task_start_outputs_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime = Runtime.open('local')
        pid = runtime.process.spawn(image='base-agent:v0', goal='object task cli')
        runtime.capability.grant(pid, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
        owner = runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {'name': 'owner'},
            metadata=ObjectMetadata(title='owner'),
            immutable=False,
        )
        monkeypatch.setattr('agent_libos.api.cli.Runtime.open', lambda *args, **kwargs: runtime)

        result = _run_cli_json([
            'object-task',
            'start',
            '--pid',
            pid,
            '--owner-oid',
            owner.oid,
            '--watch-owner',
            '--watch-events',
            'updated',
            'get_working_directory',
            '--wait',
            '--timeout',
            '2',
        ])

        assert result['status'] == 'succeeded'
        assert result['owner_oid'] == owner.oid
        assert result['owner_watch']['enabled'] is True
        assert result['owner_watch']['events'] == ['updated']
        assert result['result_oid'] is not None

    def test_cli_explain_process_lists_persisted_operations(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime = Runtime.open('local')
        pid = runtime.process.spawn(image='base-agent:v0', goal='explain cli')
        monkeypatch.setattr('agent_libos.api.cli.Runtime.open', lambda *args, **kwargs: runtime)

        result = _run_cli_json(['explain', 'process', pid])

        assert result['pid'] == pid
        assert any(operation['name'] == 'process.spawn' for operation in result['operations'])

    def test_cli_explain_not_found_uses_structured_error_and_exit_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        monkeypatch.setattr('agent_libos.api.cli.Runtime.open', lambda *args, **kwargs: runtime)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), pytest.raises(SystemExit) as raised:
            cli_main(['explain', 'operation', 'op_missing'])

        assert raised.value.code == 1
        assert json.loads(stdout.getvalue())['error']['type'] == 'NotFound'

    def test_cli_explain_ambiguous_evidence_lists_candidates_and_exits_two(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        pid = runtime.process.spawn(image='base-agent:v0', goal='ambiguous explain')
        first = runtime.operations.start(kind='runtime', name='first', actor=pid, pid=pid)
        second = runtime.operations.start(kind='runtime', name='second', actor=pid, pid=pid)
        runtime.operations.link_evidence('audit', 'shared-audit', 'audit', operation_id=first.operation_id)
        runtime.operations.link_evidence('audit', 'shared-audit', 'audit', operation_id=second.operation_id)
        runtime.operations.finish('succeeded', operation_id=first.operation_id)
        runtime.operations.finish('succeeded', operation_id=second.operation_id)
        monkeypatch.setattr('agent_libos.api.cli.Runtime.open', lambda *args, **kwargs: runtime)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), pytest.raises(SystemExit) as raised:
            cli_main(['explain', 'audit', 'shared-audit'])

        result = json.loads(stdout.getvalue())
        assert raised.value.code == 2
        assert result['ambiguous'] is True
        assert set(result['candidates']) == {first.operation_id, second.operation_id}

    def test_cli_object_task_start_requires_wait(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail_runtime_open(*_args: object, **_kwargs: object) -> None:
            raise AssertionError('argparse must reject a missing --wait before opening the Runtime')

        monkeypatch.setattr('agent_libos.api.cli.Runtime.open', fail_runtime_open)

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), pytest.raises(SystemExit) as raised:
            cli_main([
                'object-task',
                'start',
                '--pid',
                'pid-1',
                '--owner-oid',
                'oid-1',
                'get_working_directory',
            ])

        assert raised.value.code == 2
        assert 'the following arguments are required: --wait' in stderr.getvalue()

    def test_cli_object_task_start_help_describes_required_wait_and_complete_tool_table(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with pytest.raises(SystemExit) as raised:
            cli_main(['object-task', 'start', '--help'])

        assert raised.value.code == 0
        help_text = capsys.readouterr().out
        normalized_help = ' '.join(help_text.split())
        assert '[--wait]' not in help_text
        assert '--wait' in help_text
        assert 'complete process tool table' in normalized_help

    def test_cli_object_task_wait_rejects_non_finite_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime = Runtime.open('local')
        monkeypatch.setattr('agent_libos.api.cli.Runtime.open', lambda *args, **kwargs: runtime)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), pytest.raises(SystemExit) as raised:
            cli_main(['object-task', 'wait', 'task-1', '--timeout', 'nan'])

        assert '--timeout must be a finite non-negative number' in str(raised.value)

    def test_cli_object_task_watch_owner_updates_existing_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime = Runtime.open('local')
        pid = runtime.process.spawn(image='base-agent:v0', goal='object task watch cli')
        runtime.capability.grant(pid, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
        owner = runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {'name': 'owner'},
            metadata=ObjectMetadata(title='owner'),
            immutable=False,
        )
        task = runtime.object_tasks.start(pid, owner, 'receive_process_messages', {'channel': 'owner-watch'})
        runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)
        monkeypatch.setattr('agent_libos.api.cli.Runtime.open', lambda *args, **kwargs: runtime)

        result = _run_cli_json([
            'object-task',
            'watch-owner',
            task.task_id,
            '--pid',
            pid,
            '--watch-events',
            'updated',
            '--watch-channel',
            'owner-watch',
            '--watch-kind',
            'interrupt',
        ])

        assert result['task_id'] == task.task_id
        assert result['owner_watch']['enabled'] is True
        assert result['owner_watch']['events'] == ['updated']
        assert result['owner_watch']['channel'] == 'owner-watch'
        assert result['owner_watch']['kind'] == 'interrupt'

    def test_cli_mcp_register_list_inspect_and_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = root / 'runtime.sqlite'
            manifest = root / 'mcp.yaml'
            manifest.write_text(_cli_mcp_manifest('cli-mcp'), encoding='utf-8')

            registered = _run_cli_json(['--db', str(db), 'mcp', 'register', str(manifest)])
            listed = _run_cli_json(['--db', str(db), 'mcp', 'list'])
            inspected = _run_cli_json(['--db', str(db), 'mcp', 'inspect', 'cli-mcp'])
            tools = _run_cli_json(['--db', str(db), 'mcp', 'tools', 'cli-mcp'])
            second_manifest = root / 'mcp-second.yaml'
            second_manifest.write_text(_cli_mcp_manifest('cli-mcp-second'), encoding='utf-8')
            _run_cli_json(['--db', str(db), 'mcp', 'register', str(second_manifest)])
            limited = _run_cli_json(['--db', str(db), 'mcp', 'list', '--limit', '1'])
            config = root / 'config.yaml'
            config.write_text(
                'mcp:\n'
                '  list_limit: 1\n'
                '  server_page_limit: 2\n'
                '  tool_catalog_limit: 3\n',
                encoding='utf-8',
            )
            with pytest.warns(
                AgentLibOSConfigDeprecationWarning,
                match=r'mcp\.list_limit is deprecated',
            ):
                purpose_limited = _run_cli_json([
                    '--config', str(config), '--db', str(db), 'mcp', 'list',
                ])

            assert registered['server_id'] == 'cli-mcp'
            assert set(listed) == {'servers', 'has_more'}
            assert listed['has_more'] is False
            assert listed['servers'][0]['server_id'] == 'cli-mcp'
            assert inspected['transport']['type'] == 'stdio'
            assert tools['tools'][0]['tool_id'] == 'echo'
            assert tools['tools'][0]['resource'] == 'mcp:cli-mcp:echo'
            assert len(limited['servers']) == 1
            assert limited['has_more'] is True
            assert len(purpose_limited['servers']) == 2
            assert purpose_limited['has_more'] is False

    def test_cli_mcp_discover_forwards_process_actor_and_projects_connection(self) -> None:
        _parser, args = _parse_cli_args(
            [
                'mcp',
                '--actor-pid',
                'pid-modern-reader',
                'discover',
                'modern-server',
            ]
        )
        calls: list[dict[str, object]] = []
        discovery = McpDiscoveryResult(
            server_id='modern-server',
            connection=McpConnectionInfo(
                protocol_mode=McpProtocolMode.AUTO,
                protocol_era=McpProtocolEra.MODERN,
                protocol_revision='2026-07-28',
                sessionless=True,
                fallback_used=False,
                server_name='fixture',
                server_version='2.0.0',
                capabilities=('tools',),
            ),
            request_bytes=41,
            response_bytes=97,
            duration_s=0.01,
            receipts=(
                McpExchangeReceipt(
                    phase=McpExchangePhase.SERVER_DISCOVER,
                    request_bytes=41,
                    response_bytes=97,
                    duration_s=0.01,
                    call_started=True,
                ),
            ),
        )
        expected = {
            'server_id': 'modern-server',
            'connection': {
                'protocol_mode': 'auto',
                'protocol_era': 'modern',
                'protocol_revision': '2026-07-28',
                'sessionless': True,
                'fallback_used': False,
                'server_name': 'fixture',
                'server_version': '2.0.0',
                'capabilities': ['tools'],
                'unsupported_capabilities': [],
            },
            'request_bytes': 41,
            'response_bytes': 97,
            'duration_s': 0.01,
            'receipts': [
                {
                    'phase': 'server/discover',
                    'request_bytes': 41,
                    'response_bytes': 97,
                    'duration_s': 0.01,
                    'call_started': True,
                }
            ],
        }

        class RecordingMcp:
            @staticmethod
            def discover(
                server_id: str,
                *,
                actor: str,
                require_capability: bool,
            ) -> McpDiscoveryResult:
                calls.append(
                    {
                        'server_id': server_id,
                        'actor': actor,
                        'require_capability': require_capability,
                    }
                )
                return discovery

        result = _run_mcp_command(
            SimpleNamespace(mcp=RecordingMcp()),
            args,
        )

        assert result == expected
        assert result['connection']['protocol_revision'] == '2026-07-28'
        assert calls == [
            {
                'server_id': 'modern-server',
                'actor': 'pid-modern-reader',
                'require_capability': True,
            }
        ]

@contextlib.contextmanager
def _temporary_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)

def _run_cli_json(argv: list[str]) -> dict[str, object]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        cli_main(argv)
    return json.loads(stdout.getvalue())


def _cli_jsonrpc_manifest(endpoint_id: str) -> str:
    return f"""
schema_version: 1
endpoint_id: {endpoint_id}
url: https://api.example.test/jsonrpc
methods:
  - method_id: echo
    rpc_method: demo.echo
    right: read
    rollback_class: no_rollback_required
    state_mutation: false
    information_flow: true
""".lstrip()


def _cli_mcp_manifest(server_id: str) -> str:
    return f"""
schema_version: 1
server_id: {server_id}
transport: stdio
stdio:
  command: {MCP_TEST_STDIO_COMMAND_YAML}
  args: ["-m", "demo_mcp"]
tools:
  - tool_id: echo
    mcp_name: demo.echo
    right: read
    rollback_class: no_rollback_required
    state_mutation: false
    information_flow: true
timeout_s: 5
max_request_bytes: 65536
max_response_bytes: 1048576
""".strip()


def _write_cli_image_package(root: Path) -> None:
    root.mkdir(parents=True)
    root.joinpath('IMAGE.yaml').write_text("""
image_id: cli-package-agent:v0
name: cli-package-agent
prompt: prompt.md
default_tools:
  - human_output
context_policy: evidence_first
""".lstrip(), encoding='utf-8')
    root.joinpath('prompt.md').write_text('CLI loaded image.\n', encoding='utf-8')
