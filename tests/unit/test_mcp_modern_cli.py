from __future__ import annotations

import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos.api.cli import (
    _mcp_cli_operation_result,
    _mcp_cli_page,
    _parse_cli_args,
    _run_mcp_command,
    cli,
    main,
)
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.mcp.types import (
    McpAuthorizationChallenge,
    McpComplete,
    McpCompletionResult,
    McpOAuthStatus,
    McpOAuthStatusKind,
    McpPage,
    McpPromptResult,
    McpRemoteTask,
    McpRemoteTaskStatus,
    McpResource,
    McpSubscription,
    McpSubscriptionEvent,
    McpSubscriptionStatus,
)
from agent_libos.models.exceptions import ValidationError


_HUMAN_PREVIEW_SHA256 = "a" * 64
_HUMAN_REQUEST_ID = "human-request-1"


def _write_oauth_profile(path: Path, *, profile_id: str = "profile-1") -> Path:
    path.write_text(
        json.dumps(
            {
                "profile_id": profile_id,
                "server_id": "server-1",
                "resource_uri": "https://mcp.example.test/mcp",
                "expected_issuer": "https://auth.example.test/tenant",
                "redirect_uri": "https://client.example.test/oauth/callback",
                "client_id": "agent-libos-cli",
                "registration_mode": "preregistered",
            }
        ),
        encoding="utf-8",
    )
    return path


def _args(*values: str) -> Any:
    _parser, args = _parse_cli_args(["mcp", *values])
    return args


def _runtime(manager: Any) -> Any:
    return SimpleNamespace(mcp=manager, config=DEFAULT_CONFIG)


def _input_required_result() -> dict[str, Any]:
    return {
        "kind": "input_required",
        "continuation_id": "continuation-local",
        "input_requests": [
            {
                "request_id": "input-1",
                "kind": "elicitation",
                "mode": "form",
                "prompt": "Approve?",
                "schema": {
                    "type": "object",
                    "properties": {"state": {"type": "string"}},
                },
                "inert_url": None,
            }
        ],
        "expires_at": None,
        "revision": 1,
        "respondable": True,
        "human_request_id": _HUMAN_REQUEST_ID,
        "human_revision": 2,
        "human_preview_sha256": _HUMAN_PREVIEW_SHA256,
    }


def _remote_task_result() -> dict[str, Any]:
    return {
        "kind": "remote_task",
        "task_ref": "task-local",
        "status": "working",
        "status_message": None,
        "result": {"state": "domain-value"},
        "input_requests": [],
        "created_at": None,
        "updated_at": None,
        "ttl_ms": None,
        "poll_interval_ms": None,
        "revision": 1,
        "human_request_id": None,
        "human_revision": None,
        "human_preview_sha256": None,
    }


def _resource_page_result() -> dict[str, Any]:
    return {
        "items": [
            {
                "resource_id": "status",
                "name": "Status",
                "title": None,
                "description": None,
                "mime_type": "text/plain",
                "size": None,
                "icons": [],
                "annotations": None,
                "metadata": {"vendor_extension": {"state": "preserved"}},
            }
        ],
        "next_cursor": None,
        "cache_hint": {"ttl_ms": 1000, "scope": "private"},
    }


@pytest.mark.parametrize(
    "values",
    [
        ("resources", "list", "server-1"),
        ("resources", "templates", "server-1"),
        ("resources", "read", "server-1", "resource-1"),
        ("prompts", "list", "server-1"),
        ("prompts", "get", "server-1", "prompt-1"),
        (
            "prompts",
            "complete",
            "server-1",
            "prompt",
            "prompt-1",
            "argument",
            "val",
        ),
        ("auth", "status", "profile-1"),
        (
            "auth",
            "login",
            "profile-1",
            "--profile-file",
            "profile.json",
            "--callback-stdin",
        ),
        ("continuations", "inspect", "continuation-1"),
        ("remote-tasks", "get", "task-ref-1"),
        (
            "subscriptions",
            "listen",
            "server-1",
            "--filter",
            "resourcesListChanged",
            "--max-events",
            "1",
        ),
        ("validate", "manifest.yaml"),
        ("doctor", "manifest.yaml"),
        ("export",),
        ("import", "plan", "bundle.json"),
    ],
)
def test_modern_mcp_cli_surfaces_parse(values: tuple[str, ...]) -> None:
    args = _args(*values)

    assert args.mcp_command == values[0]


def test_resource_page_preserves_opaque_cursor_and_truncation() -> None:
    calls: list[tuple[str, str | None, str]] = []

    class Mcp:
        @staticmethod
        def list_resources(
            server_id: str,
            *,
            cursor: str | None,
            actor: str,
        ) -> Any:
            calls.append((server_id, cursor, actor))
            return McpPage(
                items=(McpResource(resource_id="logical-1", name="Document"),),
                next_cursor="opaque-next",
            )

    result = _run_mcp_command(
        _runtime(Mcp()),
        _args("resources", "list", "server-1", "--cursor", "opaque-current"),
    )

    assert result == {
        "resources": [
            {
                "resource_id": "logical-1",
                "name": "Document",
                "title": None,
                "description": None,
                "mime_type": None,
                "size": None,
                "icons": [],
                "annotations": None,
                "metadata": {},
            }
        ],
        "next_cursor": "opaque-next",
        "has_more": True,
        "cache_hint": None,
    }
    assert calls == [("server-1", "opaque-current", "cli")]


def test_prompt_get_is_always_an_untrusted_confirmation_preview() -> None:
    class Mcp:
        @staticmethod
        def get_prompt(
            server_id: str,
            prompt_id: str,
            *,
            arguments: dict[str, str],
            actor: str,
        ) -> Any:
            assert (server_id, prompt_id, arguments, actor) == (
                "server-1",
                "prompt-1",
                {"topic": "MCP"},
                "cli",
            )
            return McpComplete(
                value=McpPromptResult(
                    prompt_id="prompt-1",
                    messages=(),
                    user_confirmation_required=True,
                )
            )

    result = _run_mcp_command(
        _runtime(Mcp()),
        _args(
            "prompts",
            "get",
            "server-1",
            "prompt-1",
            "--arguments-json",
            '{"topic":"MCP"}',
        ),
    )

    assert result["kind"] == "complete"
    assert result["preview_only"] is True
    assert result["user_confirmation_required"] is True
    assert result["context_trust"] == "untrusted_user_context"
    assert result["system_or_developer_injection_allowed"] is False
    assert result["value"]["user_confirmation_required"] is True


def test_prompt_get_fails_closed_if_facade_waives_user_confirmation() -> None:
    class Mcp:
        @staticmethod
        def get_prompt(*_args: Any, **_kwargs: Any) -> Any:
            return McpComplete(
                value=McpPromptResult(
                    prompt_id="prompt-1",
                    messages=(),
                    user_confirmation_required=False,
                )
            )

    with pytest.raises(ValidationError, match="waived user confirmation"):
        _run_mcp_command(
            _runtime(Mcp()),
            _args("prompts", "get", "server-1", "prompt-1"),
        )


def test_completion_maps_cli_resource_template_spelling_to_pinned_api_value() -> None:
    calls: list[tuple[Any, ...]] = []

    class Mcp:
        @staticmethod
        def complete_prompt(
            server_id: str,
            reference_type: str,
            reference_id: str,
            argument: dict[str, str],
            *,
            context: dict[str, str] | None,
            actor: str,
        ) -> Any:
            calls.append(
                (server_id, reference_type, reference_id, argument, context, actor)
            )
            return McpComplete(
                value=McpCompletionResult(
                    values=("2.0.0",),
                    total=1,
                    has_more=False,
                )
            )

    result = _run_mcp_command(
        _runtime(Mcp()),
        _args(
            "prompts",
            "complete",
            "server-1",
            "resource-template",
            "release-template",
            "version",
            "2",
            "--context-json",
            '{"channel":"stable"}',
        ),
    )

    assert result["kind"] == "complete"
    assert result["preview_only"] is True
    assert result["user_confirmation_required"] is True
    assert result["system_or_developer_injection_allowed"] is False
    assert result["value"] == {
        "values": ["2.0.0"],
        "total": 1,
        "has_more": False,
    }
    assert calls == [
        (
            "server-1",
            "resource_template",
            "release-template",
            {"name": "version", "value": "2"},
            {"channel": "stable"},
            "cli",
        )
    ]


@pytest.mark.parametrize(
    ("result", "allowed_kinds"),
    [
        (
            {"kind": "complete", "value": {"state": "opaque"}, "preview_sha256": None},
            ("complete",),
        ),
        (_input_required_result(), ("input_required",)),
        (_remote_task_result(), ("remote_task",)),
    ],
)
def test_mcp_operation_envelopes_reject_unknown_top_level_fields(
    result: dict[str, Any],
    allowed_kinds: tuple[str, ...],
) -> None:
    marker = "TOP_LEVEL_PRIVATE_MARKER"
    selected = json.loads(json.dumps(result))
    selected["private_state"] = marker

    with pytest.raises(ValidationError, match="unsupported fields") as raised:
        _mcp_cli_operation_result(
            selected,
            operation="test/operation",
            allowed_kinds=allowed_kinds,
        )

    assert marker not in str(raised.value)


@pytest.mark.parametrize("kind", ["input_required", "remote_task"])
def test_mcp_input_requests_reject_unknown_fields_but_preserve_schema(
    kind: str,
) -> None:
    selected = (
        _input_required_result()
        if kind == "input_required"
        else {
            **_remote_task_result(),
            "status": "input_required",
            "input_requests": _input_required_result()["input_requests"],
            "human_request_id": _HUMAN_REQUEST_ID,
            "human_revision": 2,
            "human_preview_sha256": _HUMAN_PREVIEW_SHA256,
        }
    )
    allowed = (kind,)
    accepted = _mcp_cli_operation_result(
        selected,
        operation="test/input",
        allowed_kinds=allowed,
    )
    assert accepted["input_requests"][0]["schema"]["properties"]["state"] == {
        "type": "string"
    }

    malicious = json.loads(json.dumps(selected))
    malicious["input_requests"][0]["private_state"] = "NESTED_PRIVATE_MARKER"
    with pytest.raises(ValidationError, match="unsupported fields") as raised:
        _mcp_cli_operation_result(
            malicious,
            operation="test/input",
            allowed_kinds=allowed,
        )
    assert "NESTED_PRIVATE_MARKER" not in str(raised.value)


@pytest.mark.parametrize("unknown_location", ["page", "item", "cache"])
def test_mcp_pages_reject_unknown_page_item_and_cache_fields(
    unknown_location: str,
) -> None:
    selected = _resource_page_result()
    marker = "PAGE_PRIVATE_MARKER"
    target = {
        "page": selected,
        "item": selected["items"][0],
        "cache": selected["cache_hint"],
    }[unknown_location]
    assert isinstance(target, dict)
    target["private_state"] = marker

    with pytest.raises(ValidationError, match="unsupported fields") as raised:
        _mcp_cli_page(
            selected,
            item_key="resources",
            item_contract="resource",
            operation="resources/list",
        )

    assert marker not in str(raised.value)


def test_mcp_pages_preserve_only_explicit_metadata_extension_slots() -> None:
    selected = _resource_page_result()

    result = _mcp_cli_page(
        selected,
        item_key="resources",
        item_contract="resource",
        operation="resources/list",
    )

    assert result["resources"][0]["metadata"] == {
        "vendor_extension": {"state": "preserved"}
    }


def test_generic_complete_value_and_remote_task_result_remain_opaque_json() -> None:
    complete = _mcp_cli_operation_result(
        {
            "kind": "complete",
            "value": {"state": "domain-value"},
            "preview_sha256": None,
        },
        operation="continuations/respond",
        allowed_kinds=("complete", "input_required", "remote_task"),
    )
    task = _mcp_cli_operation_result(
        _remote_task_result(),
        operation="remote-tasks/get",
        allowed_kinds=("remote_task",),
    )

    assert complete["value"] == {"state": "domain-value"}
    assert task["result"] == {"state": "domain-value"}


def test_mcp_surface_kind_contract_rejects_generic_union_mismatch() -> None:
    with pytest.raises(ValidationError, match="invalid result kind"):
        _mcp_cli_operation_result(
            _input_required_result(),
            operation="prompts/complete",
            allowed_kinds=("complete",),
            complete_contract="completion_result",
        )


def test_completion_value_rejects_unknown_prompt_preview_fields() -> None:
    with pytest.raises(ValidationError, match="unsupported fields"):
        _mcp_cli_operation_result(
            {
                "kind": "complete",
                "value": {
                    "values": ["2.0.0"],
                    "total": 1,
                    "has_more": False,
                    "user_confirmation_required": True,
                },
                "preview_sha256": None,
            },
            operation="prompts/complete",
            allowed_kinds=("complete",),
            complete_contract="completion_result",
            prompt_preview=True,
        )


def test_resource_and_prompt_content_envelopes_reject_unknown_fields() -> None:
    text = {
        "kind": "text",
        "text": "safe",
        "annotations": None,
        "metadata": {"vendor_extension": {"state": "preserved"}},
    }
    values = (
        (
            "resource_contents",
            {
                "resource_id": "status",
                "contents": [text],
                "provenance": "untrusted_mcp_resource",
            },
        ),
        (
            "prompt_result",
            {
                "prompt_id": "review",
                "messages": [
                    {
                        "role": "user",
                        "content": text,
                        "provenance": "untrusted_mcp_prompt",
                    }
                ],
                "description": None,
                "user_confirmation_required": True,
            },
        ),
    )
    for contract, value in values:
        accepted = _mcp_cli_operation_result(
            {"kind": "complete", "value": value, "preview_sha256": None},
            operation="test/content",
            allowed_kinds=("complete",),
            complete_contract=contract,
        )
        assert accepted["value"] == value

        malicious = json.loads(json.dumps(value))
        content = (
            malicious["contents"][0]
            if contract == "resource_contents"
            else malicious["messages"][0]["content"]
        )
        content["private_state"] = "NESTED_CONTENT_MARKER"
        with pytest.raises(ValidationError, match="unsupported fields") as raised:
            _mcp_cli_operation_result(
                {
                    "kind": "complete",
                    "value": malicious,
                    "preview_sha256": None,
                },
                operation="test/content",
                allowed_kinds=("complete",),
                complete_contract=contract,
            )
        assert "NESTED_CONTENT_MARKER" not in str(raised.value)


@pytest.mark.parametrize("payload", ["null", "[]", '{"x":1}'])
def test_resource_variables_reject_non_string_mapping_before_dispatch(
    payload: str,
) -> None:
    calls: list[object] = []

    class Mcp:
        @staticmethod
        def read_resource(*args: Any, **kwargs: Any) -> Any:
            calls.append((args, kwargs))
            return McpComplete(value={})

    with pytest.raises(ValidationError):
        _run_mcp_command(
            _runtime(Mcp()),
            _args(
                "resources",
                "read",
                "server-1",
                "resource-1",
                "--variables-json",
                payload,
            ),
        )

    assert calls == []


def test_modern_json_is_bounded_before_decode_or_runtime_dispatch() -> None:
    calls: list[object] = []

    class Mcp:
        @staticmethod
        def get_prompt(*args: Any, **kwargs: Any) -> Any:
            calls.append((args, kwargs))
            return McpComplete(value={})

    runtime = SimpleNamespace(
        mcp=Mcp(),
        config=SimpleNamespace(
            mcp=SimpleNamespace(max_request_hard_limit_bytes=8),
        ),
    )
    with pytest.raises(ValidationError, match="max_bytes=8"):
        _run_mcp_command(
            runtime,
            _args(
                "prompts",
                "get",
                "server-1",
                "prompt-1",
                "--arguments-json",
                '{"value":"too-large"}',
            ),
        )

    assert calls == []


def test_continuation_response_calls_only_local_ref_revision_and_responses() -> None:
    calls: list[tuple[Any, ...]] = []

    class Mcp:
        @staticmethod
        async def respond_continuation(
            continuation_id: str,
            *,
            expected_revision: int,
            responses: dict[str, Any],
            human_request_id: str,
            human_expected_revision: int,
            human_preview_sha256: str,
            actor: str,
        ) -> Any:
            calls.append(
                (
                    continuation_id,
                    expected_revision,
                    responses,
                    human_request_id,
                    human_expected_revision,
                    human_preview_sha256,
                    actor,
                )
            )
            return McpComplete(value={"accepted": True})

    result = _run_mcp_command(
        _runtime(Mcp()),
        _args(
            "continuations",
            "respond",
            "continuation-1",
            "--expected-revision",
            "7",
            "--human-request-id",
            _HUMAN_REQUEST_ID,
            "--human-expected-revision",
            "3",
            "--human-preview-sha256",
            _HUMAN_PREVIEW_SHA256,
            "--responses-json",
            '{"input-1":{"action":"accept","content":{"approved":true}}}',
        ),
    )

    assert result["kind"] == "complete"
    assert result["value"] == {"accepted": True}
    assert calls == [
        (
            "continuation-1",
            7,
            {
                "input-1": {
                    "action": "accept",
                    "content": {"approved": True},
                }
            },
            _HUMAN_REQUEST_ID,
            3,
            _HUMAN_PREVIEW_SHA256,
            "cli",
        )
    ]


def test_remote_task_update_binds_the_human_request_receipt() -> None:
    calls: list[tuple[Any, ...]] = []

    class Mcp:
        @staticmethod
        def update_remote_task(
            task_ref: str,
            *,
            expected_revision: int,
            responses: dict[str, Any],
            human_request_id: str,
            human_expected_revision: int,
            human_preview_sha256: str,
            actor: str,
        ) -> Any:
            calls.append(
                (
                    task_ref,
                    expected_revision,
                    responses,
                    human_request_id,
                    human_expected_revision,
                    human_preview_sha256,
                    actor,
                )
            )
            return McpRemoteTask(
                task_ref="task-ref-1",
                status=McpRemoteTaskStatus.WORKING,
                revision=5,
            )

    result = _run_mcp_command(
        _runtime(Mcp()),
        _args(
            "remote-tasks",
            "update",
            "task-ref-1",
            "--expected-revision",
            "4",
            "--human-request-id",
            _HUMAN_REQUEST_ID,
            "--human-expected-revision",
            "9",
            "--human-preview-sha256",
            _HUMAN_PREVIEW_SHA256,
            "--responses-json",
            '{"input-1":{"action":"accept","content":{"approved":true}}}',
        ),
    )

    assert result["kind"] == "remote_task"
    assert result["status"] == "working"
    assert calls == [
        (
            "task-ref-1",
            4,
            {
                "input-1": {
                    "action": "accept",
                    "content": {"approved": True},
                }
            },
            _HUMAN_REQUEST_ID,
            9,
            _HUMAN_PREVIEW_SHA256,
            "cli",
        )
    ]


def test_absent_runtime_facade_fails_instead_of_returning_placeholder_success() -> None:
    with pytest.raises(ValidationError, match="operation is unavailable"):
        _run_mcp_command(
            _runtime(SimpleNamespace()),
            _args("resources", "list", "server-1"),
        )


def test_oauth_foreground_login_uses_one_runtime_and_redacts_callback_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    callback = "https://client.example/callback?code=secret-code&state=secret-state"
    profile_file = _write_oauth_profile(tmp_path / "profile.json")
    calls: list[tuple[Any, ...]] = []

    class Mcp:
        @staticmethod
        def add_oauth_profile(
            profile: Any,
            *,
            client_secret: bytes | None,
            actor: str,
        ) -> McpOAuthStatus:
            calls.append(("add", profile.profile_id, client_secret, actor))
            return McpOAuthStatus(
                profile_id=profile.profile_id,
                status=McpOAuthStatusKind.AUTHORIZATION_REQUIRED,
            )

        @staticmethod
        def auth_begin(
            profile_id: str,
            *,
            scopes: tuple[str, ...],
            actor: str,
        ) -> McpAuthorizationChallenge:
            calls.append(("begin", profile_id, scopes, actor))
            return McpAuthorizationChallenge(
                challenge_id="challenge-1",
                authorization_url="https://auth.example.test/authorize?state=public-browser-state",
                expires_at="2031-01-01T00:00:00Z",
            )

        @staticmethod
        def auth_complete(
            challenge_id: str,
            callback_url: str,
            *,
            actor: str,
        ) -> Any:
            calls.append(("complete", challenge_id, callback_url, actor))
            raise ValidationError(
                f"bad callback actor={actor} challenge={challenge_id} callback={callback_url}"
            )

    monkeypatch.setattr("sys.stdin", io.StringIO(f"{callback}\n"))
    with pytest.raises(ValidationError) as exc_info:
        _run_mcp_command(
            _runtime(Mcp()),
            _args(
                "auth",
                "login",
                "profile-1",
                "--profile-file",
                str(profile_file),
                "--scope",
                "resources.read",
                "--callback-stdin",
            ),
        )

    assert [call[0] for call in calls] == ["add", "begin", "complete"]
    assert calls[0] == ("add", "profile-1", None, "cli")
    assert calls[1] == ("begin", "profile-1", ("resources.read",), "cli")
    assert "secret-code" not in str(exc_info.value)
    assert "secret-state" not in str(exc_info.value)
    assert callback not in str(exc_info.value)
    terminal = capsys.readouterr()
    assert "secret-code" not in terminal.err
    assert "secret-state" not in terminal.err


def test_oauth_foreground_callback_failure_is_json_exit_one_without_reflection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    callback = "https://client.example/callback?code=secret-code&state=secret-state"
    profile_file = _write_oauth_profile(tmp_path / "profile.json")

    class Mcp:
        @staticmethod
        def add_oauth_profile(
            profile: Any,
            *,
            client_secret: bytes | None,
            actor: str,
        ) -> McpOAuthStatus:
            assert (profile.profile_id, client_secret, actor) == (
                "profile-1",
                None,
                "cli",
            )
            return McpOAuthStatus(
                profile_id=profile.profile_id,
                status=McpOAuthStatusKind.AUTHORIZATION_REQUIRED,
            )

        @staticmethod
        def auth_begin(
            profile_id: str,
            *,
            scopes: tuple[str, ...],
            actor: str,
        ) -> McpAuthorizationChallenge:
            assert (profile_id, scopes, actor) == ("profile-1", (), "cli")
            return McpAuthorizationChallenge(
                challenge_id="challenge-1",
                authorization_url="https://auth.example.test/authorize?state=browser-only",
                expires_at="2031-01-01T00:00:00Z",
            )

        @staticmethod
        def auth_complete(
            challenge_id: str,
            callback_url: str,
            *,
            actor: str,
        ) -> Any:
            raise ValidationError(
                f"bad callback actor={actor} challenge={challenge_id} callback={callback_url}"
            )

    runtime = SimpleNamespace(
        mcp=Mcp(),
        config=DEFAULT_CONFIG,
        shutdown=lambda **_kwargs: {"ok": True},
    )
    monkeypatch.setattr("agent_libos.api.cli.Runtime.open", lambda *_a, **_k: runtime)
    monkeypatch.setattr("sys.stdin", io.StringIO(f"{callback}\n"))

    with pytest.raises(SystemExit) as exc_info:
        cli(
            [
                "mcp",
                "auth",
                "login",
                "profile-1",
                "--profile-file",
                str(profile_file),
                "--callback-stdin",
            ]
        )

    assert exc_info.value.code == 1
    output = capsys.readouterr()
    payload = json.loads(output.out)
    assert payload["error"]["message"] == (
        "MCP auth_complete failed; sensitive request details were omitted"
    )
    encoded = output.out + output.err
    assert callback not in encoded
    assert "secret-code" not in encoded
    assert "secret-state" not in encoded


def test_oauth_foreground_login_reads_client_secret_only_from_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_file = _write_oauth_profile(tmp_path / "profile.json")
    secret = b"fd-client-secret-never-print"
    callback = "https://client.example.test/oauth/callback?code=ok&state=expected"
    observed: list[tuple[Any, ...]] = []

    class Mcp:
        @staticmethod
        def add_oauth_profile(
            profile: Any,
            *,
            client_secret: bytes | None,
            actor: str,
        ) -> McpOAuthStatus:
            observed.append(("add", profile.profile_id, client_secret, actor))
            return McpOAuthStatus(
                profile_id=profile.profile_id,
                status=McpOAuthStatusKind.AUTHORIZATION_REQUIRED,
            )

        @staticmethod
        def auth_begin(
            profile_id: str,
            *,
            scopes: tuple[str, ...],
            actor: str,
        ) -> McpAuthorizationChallenge:
            observed.append(("begin", profile_id, scopes, actor))
            return McpAuthorizationChallenge(
                challenge_id="challenge-1",
                authorization_url="https://auth.example.test/authorize",
                expires_at="2031-01-01T00:00:00Z",
            )

        @staticmethod
        def auth_complete(
            challenge_id: str,
            callback_url: str,
            *,
            actor: str,
        ) -> McpOAuthStatus:
            observed.append(("complete", challenge_id, callback_url, actor))
            return McpOAuthStatus(
                profile_id="profile-1",
                status=McpOAuthStatusKind.AUTHORIZED,
            )

    secret_file = tmp_path / "client-secret"
    secret_file.write_bytes(secret + b"\n")
    secret_fd = os.open(secret_file, os.O_RDONLY)
    monkeypatch.setattr("sys.stdin", io.StringIO(f"{callback}\n"))
    try:
        result = _run_mcp_command(
            _runtime(Mcp()),
            _args(
                "auth",
                "login",
                "profile-1",
                "--profile-file",
                str(profile_file),
                "--client-secret-fd",
                str(secret_fd),
                "--callback-stdin",
            ),
        )
    finally:
        os.close(secret_fd)

    assert result["status"] == "authorized"
    assert observed == [
        ("add", "profile-1", secret, "cli"),
        ("begin", "profile-1", (), "cli"),
        ("complete", "challenge-1", callback, "cli"),
    ]
    encoded = json.dumps(result) + capsys.readouterr().err
    assert secret.decode() not in encoded
    assert callback not in encoded


def test_oauth_foreground_login_rejects_noninteractive_stdin_without_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_file = _write_oauth_profile(tmp_path / "profile.json")
    calls: list[object] = []

    class Mcp:
        @staticmethod
        def add_oauth_profile(*args: Any, **kwargs: Any) -> Any:
            calls.append((args, kwargs))

    monkeypatch.setattr("sys.stdin", io.StringIO("callback-never-read\n"))
    with pytest.raises(ValidationError, match="TTY or explicit --callback-stdin"):
        _run_mcp_command(
            _runtime(Mcp()),
            _args(
                "auth",
                "login",
                "profile-1",
                "--profile-file",
                str(profile_file),
            ),
        )

    assert calls == []


def test_oauth_profile_file_is_exact_and_identity_bound_before_dispatch(
    tmp_path: Path,
) -> None:
    profile_file = _write_oauth_profile(
        tmp_path / "profile.json",
        profile_id="other-profile",
    )
    calls: list[object] = []

    class Mcp:
        @staticmethod
        def add_oauth_profile(*args: Any, **kwargs: Any) -> Any:
            calls.append((args, kwargs))

    with pytest.raises(ValidationError, match="identity does not match"):
        _run_mcp_command(
            _runtime(Mcp()),
            _args(
                "auth",
                "login",
                "profile-1",
                "--profile-file",
                str(profile_file),
                "--callback-stdin",
            ),
        )

    assert calls == []


def test_oauth_profile_file_rebinds_before_a_later_one_shot_status_command(
    tmp_path: Path,
) -> None:
    profile_file = _write_oauth_profile(tmp_path / "profile.json")
    calls: list[tuple[Any, ...]] = []

    class Mcp:
        @staticmethod
        def add_oauth_profile(
            profile: Any,
            *,
            client_secret: bytes | None,
            actor: str,
        ) -> McpOAuthStatus:
            calls.append(("add", profile.profile_id, client_secret, actor))
            return McpOAuthStatus(
                profile_id=profile.profile_id,
                status=McpOAuthStatusKind.AUTHORIZED,
            )

        @staticmethod
        def auth_status(profile_id: str, *, actor: str) -> McpOAuthStatus:
            calls.append(("status", profile_id, actor))
            return McpOAuthStatus(
                profile_id=profile_id,
                status=McpOAuthStatusKind.AUTHORIZED,
            )

    result = _run_mcp_command(
        _runtime(Mcp()),
        _args(
            "--oauth-profile-file",
            str(profile_file),
            "auth",
            "status",
            "profile-1",
        ),
    )

    assert result["status"] == "authorized"
    assert calls == [
        ("add", "profile-1", None, "cli"),
        ("status", "profile-1", "cli"),
    ]


def test_oauth_client_secret_fd_without_profile_fails_before_dispatch() -> None:
    calls: list[object] = []

    class Mcp:
        @staticmethod
        def list_resources(*args: Any, **kwargs: Any) -> Any:
            calls.append((args, kwargs))
            return McpPage(items=())

    with pytest.raises(ValidationError, match="requires an OAuth profile file"):
        _run_mcp_command(
            _runtime(Mcp()),
            _args(
                "--oauth-client-secret-fd",
                "3",
                "resources",
                "list",
                "server-1",
            ),
        )

    assert calls == []


def test_oauth_profile_file_rejects_embedded_secret_and_unknown_fields(
    tmp_path: Path,
) -> None:
    profile_file = _write_oauth_profile(tmp_path / "profile.json")
    selected = json.loads(profile_file.read_text(encoding="utf-8"))
    selected["client_secret"] = "must-never-be-accepted"
    profile_file.write_text(json.dumps(selected), encoding="utf-8")
    calls: list[object] = []

    class Mcp:
        @staticmethod
        def add_oauth_profile(*args: Any, **kwargs: Any) -> Any:
            calls.append((args, kwargs))

    with pytest.raises(ValidationError) as exc_info:
        _run_mcp_command(
            _runtime(Mcp()),
            _args(
                "--oauth-profile-file",
                str(profile_file),
                "auth",
                "status",
                "profile-1",
            ),
        )

    assert "must-never-be-accepted" not in str(exc_info.value)
    assert calls == []


def test_oauth_profile_is_not_bound_before_command_json_preflight(
    tmp_path: Path,
) -> None:
    profile_file = _write_oauth_profile(tmp_path / "profile.json")
    calls: list[object] = []

    class Mcp:
        @staticmethod
        def add_oauth_profile(*args: Any, **kwargs: Any) -> Any:
            calls.append((args, kwargs))

        @staticmethod
        def get_prompt(*args: Any, **kwargs: Any) -> Any:
            calls.append((args, kwargs))

    with pytest.raises(ValidationError, match="JSON object"):
        _run_mcp_command(
            _runtime(Mcp()),
            _args(
                "--oauth-profile-file",
                str(profile_file),
                "prompts",
                "get",
                "server-1",
                "prompt-1",
                "--arguments-json",
                "null",
            ),
        )

    assert calls == []


def test_oauth_profile_manifest_mismatch_fails_before_profile_effect(
    tmp_path: Path,
) -> None:
    profile_file = _write_oauth_profile(tmp_path / "profile.json")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        (Path(__file__).parents[2] / "examples" / "mcp" / "http-v3.yaml")
        .read_text(encoding="utf-8")
        + "\nauth_profile_id: profile-1\n",
        encoding="utf-8",
    )
    calls: list[object] = []

    class Mcp:
        @staticmethod
        def add_oauth_profile(*args: Any, **kwargs: Any) -> Any:
            calls.append((args, kwargs))

        @staticmethod
        def register_server_from_yaml_text(*args: Any, **kwargs: Any) -> Any:
            calls.append((args, kwargs))

    with pytest.raises(ValidationError, match="does not match the candidate"):
        _run_mcp_command(
            _runtime(Mcp()),
            _args(
                "--oauth-profile-file",
                str(profile_file),
                "register",
                str(manifest),
            ),
        )

    assert calls == []


def test_oauth_result_with_raw_token_field_fails_closed() -> None:
    class Mcp:
        @staticmethod
        def auth_status(profile_id: str, *, actor: str) -> dict[str, Any]:
            assert actor == "cli"
            return {
                "profile_id": profile_id,
                "status": "authorized",
                "access_token": "must-not-print",
            }

    with pytest.raises(ValidationError, match="forbidden secret field"):
        _run_mcp_command(
            _runtime(Mcp()),
            _args("auth", "status", "profile-1"),
        )


@pytest.mark.parametrize(
    "field",
    ["id_token", "token", "secret", "state", "client_assertion"],
)
def test_oauth_result_rejects_every_non_public_secret_field(field: str) -> None:
    class Mcp:
        @staticmethod
        def auth_status(profile_id: str, *, actor: str) -> dict[str, Any]:
            assert actor == "cli"
            return {
                "profile_id": profile_id,
                "status": "authorized",
                field: "must-not-print",
            }

    with pytest.raises(ValidationError, match="forbidden secret field"):
        _run_mcp_command(
            _runtime(Mcp()),
            _args("auth", "status", "profile-1"),
        )


def test_oauth_result_rejects_unknown_non_secret_fields_too() -> None:
    class Mcp:
        @staticmethod
        def auth_status(profile_id: str, *, actor: str) -> dict[str, Any]:
            assert actor == "cli"
            return {
                "profile_id": profile_id,
                "status": "authorized",
                "diagnostics": "provider-controlled",
            }

    with pytest.raises(ValidationError, match="unsupported OAuth status fields"):
        _run_mcp_command(
            _runtime(Mcp()),
            _args("auth", "status", "profile-1"),
        )


def test_subscription_listen_uses_one_runtime_and_stops_explicitly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[Any, ...]] = []

    class Mcp:
        @staticmethod
        def start_subscription(
            server_id: str,
            *,
            filters: tuple[str, ...],
            actor: str,
        ) -> McpSubscription:
            calls.append(("start", server_id, filters, actor))
            return McpSubscription(
                subscription_id="subscription-1",
                server_id=server_id,
                status=McpSubscriptionStatus.ACTIVE,
                requested_filters=filters,
                acknowledged_filters=filters,
            )

        @staticmethod
        def subscription_events(
            subscription_id: str,
            *,
            after: int,
            limit: int,
            actor: str,
        ) -> tuple[McpSubscriptionEvent, ...]:
            assert (subscription_id, after, limit, actor) == (
                "subscription-1",
                0,
                100,
                "cli",
            )
            calls.append(("events", subscription_id, after, limit, actor))
            return (
                McpSubscriptionEvent(
                    sequence=1,
                    event_type="resources/updated",
                    payload={"resource_id": "logical-1"},
                    received_at="2031-01-01T00:00:00Z",
                ),
            )

        @staticmethod
        def stop_subscription(
            subscription_id: str,
            *,
            actor: str,
        ) -> McpSubscription:
            calls.append(("stop", subscription_id, actor))
            return McpSubscription(
                subscription_id=subscription_id,
                server_id="server-1",
                status=McpSubscriptionStatus.CLOSED,
                requested_filters=("resourcesListChanged",),
                acknowledged_filters=("resourcesListChanged",),
            )

    result = _run_mcp_command(
        _runtime(Mcp()),
        _args(
            "subscriptions",
            "listen",
            "server-1",
            "--filter",
            "resourcesListChanged",
            "--max-events",
            "1",
        ),
    )

    assert result == {
        "subscription_id": "subscription-1",
        "events_seen": 1,
        "last_sequence": 1,
        "interrupted": False,
        "terminal_status": None,
        "stopped": {
            "subscription_id": "subscription-1",
            "server_id": "server-1",
            "status": "closed",
            "requested_filters": ["resourcesListChanged"],
            "acknowledged_filters": ["resourcesListChanged"],
            "opened_at": None,
            "closed_at": None,
            "lost_reason": None,
        },
    }
    assert calls == [
        ("start", "server-1", ("resourcesListChanged",), "cli"),
        ("events", "subscription-1", 0, 100, "cli"),
        ("stop", "subscription-1", "cli"),
    ]
    output = capsys.readouterr()
    streamed = [json.loads(line) for line in output.err.splitlines()]
    assert streamed[0]["mcp_subscription"] == "opened"
    assert streamed[1]["mcp_subscription_event"]["provenance"] == (
        "untrusted_mcp_notification"
    )


def test_subscription_listen_ctrl_c_still_stops_the_same_handle(
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[Any, ...]] = []

    class Mcp:
        @staticmethod
        def start_subscription(
            server_id: str,
            *,
            filters: tuple[str, ...],
            actor: str,
        ) -> McpSubscription:
            calls.append(("start", server_id, filters, actor))
            return McpSubscription(
                subscription_id="subscription-1",
                server_id=server_id,
                status=McpSubscriptionStatus.ACTIVE,
                requested_filters=filters,
                acknowledged_filters=filters,
            )

        @staticmethod
        def subscription_events(*_args: Any, **_kwargs: Any) -> Any:
            raise KeyboardInterrupt

        @staticmethod
        def stop_subscription(
            subscription_id: str,
            *,
            actor: str,
        ) -> McpSubscription:
            calls.append(("stop", subscription_id, actor))
            return McpSubscription(
                subscription_id=subscription_id,
                server_id="server-1",
                status=McpSubscriptionStatus.CLOSED,
                requested_filters=("resourcesListChanged",),
                acknowledged_filters=("resourcesListChanged",),
            )

    result = _run_mcp_command(
        _runtime(Mcp()),
        _args(
            "subscriptions",
            "listen",
            "server-1",
            "--filter",
            "resourcesListChanged",
        ),
    )

    assert result["interrupted"] is True
    assert result["events_seen"] == 0
    assert calls == [
        ("start", "server-1", ("resourcesListChanged",), "cli"),
        ("stop", "subscription-1", "cli"),
    ]
    capsys.readouterr()


def test_modern_host_only_surface_rejects_actor_pid_before_dispatch() -> None:
    calls: list[object] = []

    class Mcp:
        @staticmethod
        def list_prompts(*args: Any, **kwargs: Any) -> Any:
            calls.append((args, kwargs))
            return McpPage(items=())

    _parser, args = _parse_cli_args(
        ["mcp", "--actor-pid", "pid-1", "prompts", "list", "server-1"]
    )
    with pytest.raises(ValidationError, match="Host-only"):
        _run_mcp_command(_runtime(Mcp()), args)

    assert calls == []


def test_remote_tasks_intentionally_has_no_list_command() -> None:
    with pytest.raises(SystemExit) as exc_info:
        _parse_cli_args(["mcp", "remote-tasks", "list"])

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "values",
    [
        ("auth", "complete", "challenge-1"),
        ("auth", "logout", "profile-1"),
        ("subscriptions", "start", "server-1"),
        ("subscriptions", "status", "subscription-1"),
        ("subscriptions", "events", "subscription-1"),
        ("subscriptions", "stop", "subscription-1"),
    ],
)
def test_cli_rejects_cross_process_oauth_and_subscription_lifecycles(
    values: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _parse_cli_args(["mcp", *values])

    assert exc_info.value.code == 2


def test_cli_documentation_only_promises_reachable_foreground_lifecycles() -> None:
    documentation = (Path(__file__).parents[2] / "docs" / "cli.md").read_text(
        encoding="utf-8"
    )

    assert "mcp auth complete" not in documentation
    assert "mcp subscriptions start" not in documentation
    assert "mcp subscriptions status" not in documentation
    assert "mcp subscriptions events" not in documentation
    assert "mcp subscriptions stop" not in documentation
    assert "mcp --oauth-profile-file" in documentation
    assert "mcp subscriptions listen" in documentation
    assert "there is deliberately no standalone `auth complete`" in documentation
    assert '"input-1":{"action":"accept","content":{"approved":true}}' in documentation
    assert '"request-1":{"answer":"yes"}' not in documentation


@pytest.mark.parametrize(
    "values",
    [
        (
            "continuations",
            "respond",
            "continuation-1",
            "--expected-revision",
            "1",
            "--human-request-id",
            _HUMAN_REQUEST_ID,
            "--human-expected-revision",
            "1",
            "--human-preview-sha256",
            _HUMAN_PREVIEW_SHA256,
            "--responses-json",
            "{}",
            "--binding-json",
            "{}",
        ),
        (
            "remote-tasks",
            "get",
            "task-ref-1",
            "--server-id",
            "attacker-selected",
        ),
        (
            "subscriptions",
            "listen",
            "server-1",
            "--filter",
            "resourcesListChanged",
            "--last-event-id",
            "replay-me",
        ),
        (
            "import",
            "apply",
            "bundle.json",
            "server-1",
            "--confirm-import",
            "--reviewer",
            "host",
            "--reason",
            "reviewed",
            "--actor",
            "forged-audit-actor",
        ),
    ],
)
def test_modern_cli_rejects_internal_binding_and_replay_inputs(
    values: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _parse_cli_args(["mcp", *values])

    assert exc_info.value.code == 2


def test_import_apply_help_does_not_offer_an_audit_actor_override(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _parse_cli_args(["mcp", "import", "apply", "--help"])

    output = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "--actor ACTOR" not in output.out
    assert "--reviewer REVIEWER" in output.out


def test_import_apply_uses_the_fixed_cli_audit_actor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_libos.mcp import dx

    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}", encoding="utf-8")
    observed: list[tuple[Any, ...]] = []

    def apply(
        _adapter: Any,
        raw: bytes,
        *,
        server_id: str,
        confirmation: Any,
        actor: str,
        require_capability: bool,
    ) -> dict[str, Any]:
        observed.append(
            (
                raw,
                server_id,
                confirmation.actor,
                confirmation.reason,
                actor,
                require_capability,
            )
        )
        return {"server_id": server_id, "action": "unchanged"}

    monkeypatch.setattr(dx, "import_one_from_bundle", apply)
    manager = SimpleNamespace(config=SimpleNamespace(mcp=DEFAULT_CONFIG.mcp))

    result = _run_mcp_command(
        _runtime(manager),
        _args(
            "import",
            "apply",
            str(bundle),
            "server-1",
            "--confirm-import",
            "--reviewer",
            "host-reviewer",
            "--reason",
            "reviewed",
        ),
    )

    assert result == {"server_id": "server-1", "action": "unchanged"}
    assert observed == [
        (
            b"{}",
            "server-1",
            "host-reviewer",
            "reviewed",
            "mcp-cli-import",
            False,
        )
    ]


def test_dx_validate_is_wired_to_bounded_host_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_libos.mcp import dx

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("schema_version: 3\n", encoding="utf-8")
    observed: list[str] = []

    class Report:
        @staticmethod
        def to_jsonable() -> dict[str, Any]:
            return {"valid": True}

    def validate(_adapter: Any, text: str) -> Report:
        observed.append(text)
        return Report()

    monkeypatch.setattr(dx, "validate_manifest_text", validate)
    manager = SimpleNamespace(config=SimpleNamespace(mcp=DEFAULT_CONFIG.mcp))

    result = _run_mcp_command(
        _runtime(manager),
        _args("validate", str(manifest)),
    )

    assert result == {"valid": True}
    assert observed == ["schema_version: 3\n"]


def test_dx_probe_uses_the_governed_unregistered_candidate_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_libos.mcp import dx

    manifest = tmp_path / "candidate.yaml"
    manifest.write_text("schema_version: 3\n", encoding="utf-8")
    observed: list[tuple[Any, ...]] = []

    class CandidateAdapter:
        def __init__(self, adapter: Any) -> None:
            self.adapter = adapter

    class Report:
        @staticmethod
        def to_jsonable() -> dict[str, Any]:
            return {"catalog_scope": "full_catalog", "complete": True}

    def probe(
        adapter: Any,
        text: str,
        *,
        probe_adapter: Any,
        confirmation: Any,
    ) -> Report:
        observed.append(
            (
                adapter,
                text,
                probe_adapter.adapter,
                confirmation.actor,
                confirmation.reason,
                confirmation.confirmed,
            )
        )
        return Report()

    monkeypatch.setattr(dx, "CandidateMcpProbeAdapter", CandidateAdapter)
    monkeypatch.setattr(dx, "probe_manifest", probe)
    manager = SimpleNamespace(config=SimpleNamespace(mcp=DEFAULT_CONFIG.mcp))

    result = _run_mcp_command(
        _runtime(manager),
        _args(
            "probe",
            str(manifest),
            "--confirm-probe",
            "--reviewer",
            "host-reviewer",
            "--reason",
            "onboard candidate",
        ),
    )

    assert result == {"catalog_scope": "full_catalog", "complete": True}
    adapter, text, candidate_adapter, reviewer, reason, confirmed = observed[0]
    assert candidate_adapter is adapter
    assert text == "schema_version: 3\n"
    assert (reviewer, reason, confirmed) == (
        "host-reviewer",
        "onboard candidate",
        True,
    )


def test_dx_scaffold_forwards_every_probe_catalog_and_manifest_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_libos.mcp import dx

    base = tmp_path / "base.yaml"
    base.write_text("schema_version: 3\n", encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "manifest_sha256": _HUMAN_PREVIEW_SHA256,
                "catalog_scope": "full_catalog",
                "complete": True,
                "tools": [{"name": "demo.echo"}],
                "resources": [{"uri": "demo://status"}],
                "resource_templates": [
                    {"uri_template": "demo://greeting/{name}"}
                ],
                "prompts": [{"name": "demo.review"}],
            }
        ),
        encoding="utf-8",
    )
    observed: list[dict[str, Any]] = []

    def scaffold(
        _adapter: Any,
        base_manifest: dict[str, Any],
        tools: list[Any],
        **keywords: Any,
    ) -> dict[str, Any]:
        observed.append(
            {
                "base": base_manifest,
                "tools": tools,
                **keywords,
            }
        )
        return {"kind": "agent-libos.mcp-manifest-candidate"}

    monkeypatch.setattr(dx, "scaffold_manifest_candidate", scaffold)
    manager = SimpleNamespace(config=SimpleNamespace(mcp=DEFAULT_CONFIG.mcp))

    result = _run_mcp_command(
        _runtime(manager),
        _args(
            "scaffold",
            "create",
            str(base),
            str(catalog),
            "--confirm-scaffold",
            "--reviewer",
            "host-reviewer",
            "--reason",
            "review all catalogs",
        ),
    )

    assert result == {"kind": "agent-libos.mcp-manifest-candidate"}
    assert observed[0]["tools"] == [{"name": "demo.echo"}]
    assert observed[0]["live_resources"] == [{"uri": "demo://status"}]
    assert observed[0]["live_resource_templates"] == [
        {"uri_template": "demo://greeting/{name}"}
    ]
    assert observed[0]["live_prompts"] == [{"name": "demo.review"}]
    assert observed[0]["probe_manifest_sha256"] == _HUMAN_PREVIEW_SHA256
    assert observed[0]["catalog_scope"] == "full_catalog"
    assert observed[0]["complete"] is True


@pytest.mark.parametrize(
    "values",
    [
        ("validate", "manifest.yaml"),
        ("doctor", "manifest.yaml"),
        (
            "scaffold",
            "approve",
            "candidate.json",
            "--confirm-review",
            "--reviewer",
            "host",
            "--reason",
            "reviewed",
        ),
    ],
)
def test_offline_dx_commands_never_open_the_selected_persistent_store(
    values: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str | None] = []
    dispatched: list[tuple[Any, str]] = []
    runtime = object()

    def open_runtime(target: str | None, **_kwargs: Any) -> object:
        opened.append(target)
        return runtime

    monkeypatch.setattr("agent_libos.api.cli.Runtime.open", open_runtime)
    monkeypatch.setattr(
        "agent_libos.api.cli._dispatch_cli_command_with_shutdown",
        lambda selected, _args, display: dispatched.append((selected, display)),
    )

    main(["--db", "must-not-open.sqlite", "mcp", *values])

    assert opened == [":memory:"]
    assert dispatched == [(runtime, ":memory:")]
