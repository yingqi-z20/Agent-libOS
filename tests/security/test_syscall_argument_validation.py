from __future__ import annotations

import asyncio
import ast
import inspect
from pathlib import Path
import tempfile
import textwrap
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.mcp import (
    McpComplete,
    McpPage,
    McpResource,
    McpResourceContents,
    McpResourceTemplate,
    McpTextContent,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.runtime.syscall_descriptors import BUILTIN_SYSCALL_DESCRIPTORS
from agent_libos.runtime.syscalls import (
    _BUILTIN_SYSCALL_ARGUMENT_RULES,
    _UNVALIDATED_BUILTIN_SYSCALLS,
    LibOSSyscallSession,
)
from agent_libos.substrate import LocalResourceProviderSubstrate
from agent_libos.utils.serde import dumps, to_jsonable


_SINK_CASES = [
    pytest.param(
        "filesystem.write_text",
        {"path": "out.txt", "content": "safe", "overwrite": "false"},
        "filesystem",
        "write_text",
        "overwrite",
        id="filesystem-overwrite-string",
    ),
    pytest.param(
        "filesystem.write_directory",
        {"path": "out", "parents": 1},
        "filesystem",
        "write_directory",
        "parents",
        id="filesystem-parents-integer",
    ),
    pytest.param(
        "filesystem.write_directory",
        {"path": "out", "exist_ok": "false"},
        "filesystem",
        "write_directory",
        "exist_ok",
        id="filesystem-exist-ok-string",
    ),
    pytest.param(
        "filesystem.delete_file",
        {"path": "victim", "missing_ok": "false"},
        "filesystem",
        "delete_file",
        "missing_ok",
        id="filesystem-file-missing-ok-string",
    ),
    pytest.param(
        "filesystem.delete_directory",
        {"path": "victim", "recursive": "false"},
        "filesystem",
        "delete_directory",
        "recursive",
        id="filesystem-recursive-string",
    ),
    pytest.param(
        "filesystem.delete_directory",
        {"path": "victim", "missing_ok": 1},
        "filesystem",
        "delete_directory",
        "missing_ok",
        id="filesystem-directory-missing-ok-integer",
    ),
    pytest.param(
        "mcp.tools",
        {"server_id": "registered", "refresh": "false"},
        "mcp",
        "alist_tools",
        "refresh",
        id="mcp-refresh-string",
    ),
    pytest.param(
        "process.list_children",
        {"include_terminal": "false"},
        "process",
        "list_children",
        "include_terminal",
        id="process-include-terminal-string",
    ),
    pytest.param(
        "process.merge_child_memory",
        {"child_pid": "child", "include_child_created": 0},
        "process",
        "merge_child_memory",
        "include_child_created",
        id="process-include-child-created-integer",
    ),
    pytest.param(
        "process.wait",
        {"child_pid": "child", "block": "false"},
        "process",
        "wait",
        "block",
        id="process-block-string",
    ),
    pytest.param(
        "process.read_messages",
        {"include_acked": "false"},
        "messages",
        "receive",
        "include_acked",
        id="message-include-acked-string",
    ),
    pytest.param(
        "process.receive_messages",
        {"block": 0},
        "messages",
        "receive",
        "block",
        id="message-block-integer",
    ),
    pytest.param(
        "process.read_messages",
        {"ack": "false"},
        "messages",
        "receive",
        "ack",
        id="message-ack-string",
    ),
    pytest.param(
        "skill.register_path",
        {"path": "skill", "replace": "false"},
        "skills",
        "register_skill_from_workspace_path",
        "replace",
        id="skill-replace-string",
    ),
    pytest.param(
        "image.commit_checkpoint",
        {
            "checkpoint_id": "checkpoint",
            "image_id": "image:v0",
            "name": "image",
            "replace": "false",
        },
        "image_registry",
        "commit_from_checkpoint",
        "replace",
        id="checkpoint-image-replace-string",
    ),
    pytest.param(
        "image.load_package",
        {"path": "image", "replace": 1},
        "image_registry",
        "register_from_workspace_package",
        "replace",
        id="image-package-replace-integer",
    ),
    pytest.param(
        "memory.create_object",
        {"payload": {}, "immutable": "false"},
        "memory",
        "create_object",
        "immutable",
        id="memory-immutable-string",
    ),
    pytest.param(
        "capability.list",
        {"include_inactive": "false"},
        "capability",
        "list_subject",
        "include_inactive",
        id="capability-include-inactive-string",
    ),
    pytest.param(
        "process.fork",
        {"goal": "child", "include_parent_roots": "false"},
        "runtime",
        "fork_child_process",
        "include_parent_roots",
        id="fork-include-parent-roots-string",
    ),
    pytest.param(
        "filesystem.read_text",
        {"path": "input", "max_bytes": True},
        "filesystem",
        "read_text",
        "max_bytes",
        id="filesystem-max-bytes-boolean",
    ),
    pytest.param(
        "filesystem.read_directory",
        {"path": ".", "limit": "10"},
        "filesystem",
        "read_directory",
        "limit",
        id="filesystem-limit-string",
    ),
    pytest.param(
        "memory.list_namespace",
        {"limit": True},
        "memory",
        "list_namespace",
        "limit",
        id="memory-limit-boolean",
    ),
    pytest.param(
        "checkpoint.list",
        {"limit": "10"},
        "checkpoint",
        "list",
        "limit",
        id="checkpoint-limit-string",
    ),
    pytest.param(
        "skill.discover",
        {"limit": True},
        "skills",
        "discover_skills",
        "limit",
        id="skill-limit-boolean",
    ),
    pytest.param(
        "skill.read_resource",
        {"skill_id": "skill", "path": "SKILL.md", "max_bytes": "10"},
        "skills",
        "read_skill_resource",
        "max_bytes",
        id="skill-max-bytes-string",
    ),
    pytest.param(
        "capability.list",
        {"limit": True},
        "capability",
        "list_subject",
        "limit",
        id="capability-limit-boolean",
    ),
    pytest.param(
        "process.read_messages",
        {"limit": "10"},
        "messages",
        "receive",
        "limit",
        id="message-limit-string",
    ),
    pytest.param(
        "clock.sleep",
        {"seconds": True},
        "clock",
        "asleep",
        "seconds",
        id="sleep-seconds-boolean",
    ),
    pytest.param(
        "shell.run",
        {"argv": ["pwd"], "timeout_s": "1"},
        "shell",
        "arun",
        "timeout_s",
        id="shell-timeout-string",
    ),
    pytest.param(
        "filesystem.delete_file",
        {"path": True},
        "filesystem",
        "delete_file",
        "path",
        id="filesystem-path-boolean",
    ),
    pytest.param(
        "checkpoint.restore",
        {"checkpoint_id": 42},
        "checkpoint",
        "restore",
        "checkpoint_id",
        id="checkpoint-id-integer",
    ),
    pytest.param(
        "process.signal",
        {"child_pid": 42, "signal": "pause"},
        "process",
        "signal_child",
        "child_pid",
        id="child-pid-integer",
    ),
    pytest.param(
        "mcp.call",
        {"server_id": 42, "tool_id": "tool"},
        "mcp",
        "acall_tool",
        "server_id",
        id="mcp-server-id-integer",
    ),
]


def test_argument_contract_inventory_covers_every_builtin_except_delegate() -> None:
    canonical_names = {descriptor.name for descriptor in BUILTIN_SYSCALL_DESCRIPTORS}

    assert _UNVALIDATED_BUILTIN_SYSCALLS == {"capability.delegate"}
    assert set(_BUILTIN_SYSCALL_ARGUMENT_RULES) == (
        canonical_names - _UNVALIDATED_BUILTIN_SYSCALLS
    )


def test_argument_contract_covers_every_direct_handler_argument_key() -> None:
    missing: dict[str, list[str]] = {}
    for descriptor in BUILTIN_SYSCALL_DESCRIPTORS:
        if descriptor.name in _UNVALIDATED_BUILTIN_SYSCALLS:
            continue
        source = textwrap.dedent(
            inspect.getsource(getattr(LibOSSyscallSession, descriptor.handler))
        )
        tree = ast.parse(source)
        keys: set[str] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == "args"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                keys.add(node.slice.value)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "args"
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                keys.add(node.args[0].value)
        uncovered = sorted(
            keys - set(_BUILTIN_SYSCALL_ARGUMENT_RULES[descriptor.name])
        )
        if uncovered:
            missing[descriptor.name] = uncovered

    assert missing == {}


@pytest.mark.parametrize(
    ("syscall_name", "args", "owner_name", "method_name", "field"),
    _SINK_CASES,
)
def test_sensitive_syscall_arguments_reject_type_confusion_before_sink(
    monkeypatch: pytest.MonkeyPatch,
    syscall_name: str,
    args: dict[str, Any],
    owner_name: str,
    method_name: str,
    field: str,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="validate syscall args")
        owner = runtime if owner_name == "runtime" else getattr(runtime, owner_name)
        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def unexpected_sink(*call_args: Any, **call_kwargs: Any) -> Any:
            calls.append((call_args, call_kwargs))
            raise AssertionError("syscall reached its primitive before argument validation")

        monkeypatch.setattr(owner, method_name, unexpected_sink)
        session = LibOSSyscallSession(runtime, pid)

        with pytest.raises(
            ValidationError,
            match=rf"syscall argument '{field}' must be",
        ):
            asyncio.run(session.handle(syscall_name, args))

        assert calls == []
        assert not [
            record
            for record in runtime.audit.trace(actor=pid)
            if record.action == "syscall.result" and record.target == syscall_name
        ]
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("syscall_name", "args", "deferred_attribute", "field"),
    [
        pytest.param(
            "process.exec",
            {"image": "base-agent:v0", "preserve_memory": "false"},
            "_deferred_exec",
            "preserve_memory",
            id="exec-preserve-memory-string",
        ),
        pytest.param(
            "process.exec",
            {"image": "base-agent:v0", "preserve_capabilities": 1},
            "_deferred_exec",
            "preserve_capabilities",
            id="exec-preserve-capabilities-integer",
        ),
        pytest.param(
            "process.exit",
            {"failed": "false"},
            "_deferred_exit",
            "failed",
            id="exit-failed-string",
        ),
        pytest.param(
            "process.exit",
            {"use_tool_result": 0},
            "_deferred_exit",
            "use_tool_result",
            id="exit-use-result-integer",
        ),
    ],
)
def test_lifecycle_syscall_type_confusion_rejects_before_deferred_state(
    syscall_name: str,
    args: dict[str, Any],
    deferred_attribute: str,
    field: str,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="validate lifecycle args")
        session = LibOSSyscallSession(runtime, pid)

        with pytest.raises(
            ValidationError,
            match=rf"syscall argument '{field}' must be a boolean",
        ):
            asyncio.run(session.handle(syscall_name, args))

        assert getattr(session, deferred_attribute) is None
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("syscall_name", "args", "owner_name", "method_name", "error"),
    [
        pytest.param(
            "filesystem.read_text",
            {
                "path": "input",
                "max_bytes": DEFAULT_CONFIG.tools.filesystem_read_hard_limit_bytes
                + 1,
            },
            "filesystem",
            "read_text",
            "max_bytes' exceeds maximum",
            id="filesystem-max-bytes-hard-limit",
        ),
        pytest.param(
            "filesystem.read_directory",
            {"path": ".", "limit": 0},
            "filesystem",
            "read_directory",
            "limit' must be >= 1",
            id="filesystem-limit-positive",
        ),
        pytest.param(
            "memory.list_namespace",
            {"limit": DEFAULT_CONFIG.memory.query_limit + 1},
            "memory",
            "list_namespace",
            "limit' exceeds maximum",
            id="memory-query-limit",
        ),
        pytest.param(
            "checkpoint.list",
            {"limit": DEFAULT_CONFIG.checkpoint.list_limit + 1},
            "checkpoint",
            "list",
            "limit' exceeds maximum",
            id="checkpoint-list-limit",
        ),
        pytest.param(
            "skill.discover",
            {"limit": DEFAULT_CONFIG.skills.discover_limit + 1},
            "skills",
            "discover_skills",
            "limit' exceeds maximum",
            id="skill-discover-limit",
        ),
        pytest.param(
            "skill.read_resource",
            {
                "skill_id": "skill",
                "path": "SKILL.md",
                "max_bytes": DEFAULT_CONFIG.skills.resource_read_max_bytes + 1,
            },
            "skills",
            "read_skill_resource",
            "max_bytes' exceeds maximum",
            id="skill-resource-limit",
        ),
        pytest.param(
            "capability.list",
            {"limit": DEFAULT_CONFIG.capability.list_limit + 1},
            "capability",
            "list_subject",
            "limit' exceeds maximum",
            id="capability-list-limit",
        ),
        pytest.param(
            "process.read_messages",
            {"limit": DEFAULT_CONFIG.tools.message_read_hard_limit + 1},
            "messages",
            "receive",
            "limit' exceeds maximum",
            id="message-read-limit",
        ),
        pytest.param(
            "clock.sleep",
            {"seconds": DEFAULT_CONFIG.tools.max_sleep_seconds + 1},
            "clock",
            "asleep",
            "seconds' exceeds maximum",
            id="clock-sleep-limit",
        ),
        pytest.param(
            "shell.run",
            {"argv": ["pwd"], "timeout_s": 0},
            "shell",
            "arun",
            "timeout_s' must be > 0",
            id="shell-timeout-positive",
        ),
        pytest.param(
            "shell.run",
            {
                "argv": ["pwd"],
                "timeout_s": DEFAULT_CONFIG.shell.timeout_hard_limit_s + 1,
            },
            "shell",
            "arun",
            "timeout_s' exceeds maximum",
            id="shell-timeout-hard-limit",
        ),
    ],
)
def test_syscall_resource_bounds_reject_before_primitive(
    monkeypatch: pytest.MonkeyPatch,
    syscall_name: str,
    args: dict[str, Any],
    owner_name: str,
    method_name: str,
    error: str,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="bound syscall args")
        owner = runtime if owner_name == "runtime" else getattr(runtime, owner_name)
        calls: list[object] = []

        def unexpected_sink(*_args: Any, **_kwargs: Any) -> Any:
            calls.append(object())
            raise AssertionError("bounded syscall reached primitive")

        monkeypatch.setattr(owner, method_name, unexpected_sink)

        with pytest.raises(ValidationError, match=error):
            asyncio.run(LibOSSyscallSession(runtime, pid).handle(syscall_name, args))

        assert calls == []
    finally:
        runtime.close()


def test_recursive_string_false_cannot_delete_nonempty_directory() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        victim = root / "victim"
        victim.mkdir()
        (victim / "keep.txt").write_text("keep", encoding="utf-8")
        runtime = Runtime.open(
            "local",
            substrate=LocalResourceProviderSubstrate(root),
        )
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="keep directory")
            runtime.filesystem.grant_path_list(
                pid,
                delete_dirs=["victim"],
                issued_by="test",
            )
            session = LibOSSyscallSession(runtime, pid)

            with pytest.raises(
                ValidationError,
                match="syscall argument 'recursive' must be a boolean",
            ):
                asyncio.run(
                    session.handle(
                        "filesystem.delete_directory",
                        {"path": "victim", "recursive": "false"},
                    )
                )

            assert victim.is_dir()
            assert (victim / "keep.txt").read_text(encoding="utf-8") == "keep"
        finally:
            runtime.close()


def test_string_false_ack_cannot_ack_an_unread_message() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="keep message unread")
        message = runtime.human.send_process_message(pid, "do not acknowledge")
        session = LibOSSyscallSession(runtime, pid)

        with pytest.raises(
            ValidationError,
            match="syscall argument 'ack' must be a boolean",
        ):
            asyncio.run(session.handle("process.read_messages", {"ack": "false"}))

        assert [item.message_id for item in runtime.messages.unread(pid)] == [
            message.message_id
        ]
    finally:
        runtime.close()


def test_valid_boolean_and_number_syscall_arguments_reach_primitive_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="valid syscall args")
        observed: dict[str, Any] = {}

        def delete_directory(*_args: Any, **kwargs: Any) -> dict[str, Any]:
            observed.update(kwargs)
            return {"deleted": False}

        monkeypatch.setattr(runtime.filesystem, "delete_directory", delete_directory)
        session = LibOSSyscallSession(runtime, pid)

        result = asyncio.run(
            session.handle(
                "filesystem.delete_directory",
                {"path": "victim", "recursive": False, "missing_ok": True},
            )
        )
        asyncio.run(
            session.handle(
                "process.exec",
                {
                    "image": "base-agent:v0",
                    "preserve_memory": False,
                    "preserve_capabilities": True,
                },
            )
        )

        assert result == {"deleted": False}
        assert observed["recursive"] is False
        assert observed["missing_ok"] is True
        assert session._deferred_exec is not None
        assert session._deferred_exec["preserve_memory"] is False
        assert session._deferred_exec["preserve_capabilities"] is True
    finally:
        runtime.close()


def test_integer_and_float_timeouts_remain_valid_json_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="valid numeric args")
        observed: list[float] = []

        async def shell_run(
            _pid: str,
            _argv: list[str],
            *,
            timeout: float,
            cwd: str,
        ) -> dict[str, Any]:
            del cwd
            observed.append(timeout)
            return {"ok": True}

        monkeypatch.setattr(runtime.shell, "arun", shell_run)
        session = LibOSSyscallSession(runtime, pid)

        asyncio.run(session.handle("shell.run", {"argv": ["pwd"], "timeout_s": 1}))
        asyncio.run(
            session.handle("shell.run", {"argv": ["pwd"], "timeout_s": 1.5})
        )

        assert observed == [1.0, 1.5]
    finally:
        runtime.close()


def test_builtin_alias_uses_same_argument_contract_before_primitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="validate alias")
        called = False

        def unexpected_write(*_args: Any, **_kwargs: Any) -> Any:
            nonlocal called
            called = True
            raise AssertionError("alias reached primitive")

        monkeypatch.setattr(runtime.filesystem, "write_text", unexpected_write)

        with pytest.raises(
            ValidationError,
            match="syscall argument 'overwrite' must be a boolean",
        ):
            asyncio.run(
                LibOSSyscallSession(runtime, pid).handle(
                    "filesystem.write_text_file",
                    {"path": "out.txt", "content": "x", "overwrite": "false"},
                )
            )

        assert called is False
    finally:
        runtime.close()


def test_mcp_syscalls_reject_unknown_arguments_before_primitive_and_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "opaque-mcp-unknown-field-value-48291"
    runtime = Runtime.open("local")
    calls: list[str] = []

    def unexpected_sync(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("sync")
        raise AssertionError("unknown MCP syscall fields reached the primitive")

    async def unexpected_async(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("async")
        raise AssertionError("unknown MCP syscall fields reached the primitive")

    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="reject unknown MCP syscall arguments",
        )
        monkeypatch.setattr(runtime.mcp, "list_servers", unexpected_sync)
        monkeypatch.setattr(runtime.mcp, "inspect_server", unexpected_sync)
        monkeypatch.setattr(runtime.mcp, "alist_tools", unexpected_async)
        monkeypatch.setattr(runtime.mcp, "acall_tool", unexpected_async)
        monkeypatch.setattr(
            runtime.mcp,
            "alist_resources",
            unexpected_async,
            raising=False,
        )
        monkeypatch.setattr(
            runtime.mcp,
            "alist_resource_templates",
            unexpected_async,
            raising=False,
        )
        monkeypatch.setattr(
            runtime.mcp,
            "aread_resource",
            unexpected_async,
            raising=False,
        )
        session = LibOSSyscallSession(runtime, pid)

        cases = (
            ("mcp.list", {}),
            ("mcp.inspect", {"server_id": "registered"}),
            (
                "mcp.tools",
                {"server_id": "registered", "refresh": False},
            ),
            (
                "mcp.call",
                {
                    "server_id": "registered",
                    "tool_id": "allowed",
                    "arguments": {},
                },
            ),
            (
                "mcp.resources",
                {"server_id": "registered", "kind": "resource"},
            ),
            (
                "mcp.resource_read",
                {
                    "server_id": "registered",
                    "resource_id": "status",
                    "variables": {},
                },
            ),
        )
        for name, valid_args in cases:
            with pytest.raises(
                ValidationError,
                match="MCP syscall arguments contain unknown fields",
            ) as raised:
                asyncio.run(
                    session.handle(
                        name,
                        {
                            **valid_args,
                            "ad_hoc_transport_material": sentinel,
                        },
                    )
                )
            assert sentinel not in str(raised.value)

        assert calls == []
        assert runtime.store.list_external_effects(pid=pid) == []
        request_audits = [
            record
            for record in runtime.audit.trace(actor=pid)
            if record.action == "syscall.request"
            and record.target.startswith("mcp.")
        ]
        assert len(request_audits) == len(cases)
        assert all(
            record.decision == {
                "args": {
                    "redacted": True,
                    "argument_count": len(valid_args) + 1,
                },
                "validation": "rejected",
            }
            for record, (_name, valid_args) in zip(request_audits, cases)
        )
        persisted = dumps(
            {
                "audit": [to_jsonable(record) for record in runtime.audit.trace()],
                "events": [to_jsonable(event) for event in runtime.events.list()],
                "effects": [
                    to_jsonable(effect)
                    for effect in runtime.store.list_external_effects(pid=pid)
                ],
            }
        )
        assert sentinel not in persisted
    finally:
        runtime.close()


def test_mcp_call_syscall_keeps_nested_tool_arguments_schema_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    observed: list[dict[str, Any]] = []

    async def capture_call(
        _pid: str,
        *,
        server_id: str,
        tool_id: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        assert server_id == "registered"
        assert tool_id == "allowed"
        observed.append(arguments)
        return {"ok": True}

    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="preserve nested MCP tool arguments",
        )
        monkeypatch.setattr(runtime.mcp, "acall_tool", capture_call)
        result = asyncio.run(
            LibOSSyscallSession(runtime, pid).handle(
                "mcp.call",
                {
                    "server_id": "registered",
                    "tool_id": "allowed",
                    "arguments": {"server_defined_field": {"nested": True}},
                },
            )
        )

        assert result == {"ok": True}
        assert observed == [{"server_defined_field": {"nested": True}}]
    finally:
        runtime.close()


def test_mcp_call_syscall_rejects_explicit_null_arguments_before_primitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    called = False

    async def unexpected_call(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("explicit null MCP arguments reached the primitive")

    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="reject explicit null MCP syscall arguments",
        )
        monkeypatch.setattr(runtime.mcp, "acall_tool", unexpected_call)
        with pytest.raises(
            ValidationError,
            match="syscall argument 'arguments' must be an object",
        ):
            asyncio.run(
                LibOSSyscallSession(runtime, pid).handle(
                    "mcp.call",
                    {
                        "server_id": "registered",
                        "tool_id": "allowed",
                        "arguments": None,
                    },
                )
            )
        assert called is False
        assert runtime.store.list_external_effects(pid=pid) == []
    finally:
        runtime.close()


def test_mcp_list_syscall_preserves_truncation_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="observe bounded MCP registry list",
        )
        monkeypatch.setattr(
            runtime.mcp,
            "list_servers_window",
            lambda **_kwargs: ([{"server_id": "visible"}], True),
        )

        result = asyncio.run(
            LibOSSyscallSession(runtime, pid).handle("mcp.list", {})
        )

        assert result == {
            "servers": [{"server_id": "visible"}],
            "has_more": True,
        }
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("name", "arguments", "message"),
    [
        (
            "mcp.resources",
            {"server_id": "https://attacker.invalid"},
            "logical id is invalid",
        ),
        (
            "mcp.resources",
            {"server_id": "registered", "cursor": "provider-raw-cursor"},
            "cursor must be opaque",
        ),
        (
            "mcp.resources",
            {"server_id": "registered", "kind": "prompt"},
            "kind is invalid",
        ),
        (
            "mcp.resource_read",
            {
                "server_id": "registered",
                "resource_id": "file:///host-secret",
            },
            "logical id is invalid",
        ),
        (
            "mcp.resource_read",
            {
                "server_id": "registered",
                "resource_id": "status",
                "variables": {"name": 7},
            },
            "logical string names and values",
        ),
        (
            "mcp.resource_read",
            {
                "server_id": "registered",
                "resource_id": "status",
                "variables": {"bad/key": "value"},
            },
            "logical string names and values",
        ),
        (
            "mcp.resource_read",
            {
                "server_id": "registered",
                "resource_id": "status",
                "actor": "attacker",
            },
            "unknown fields",
        ),
    ],
    ids=(
        "server-url",
        "raw-provider-cursor",
        "prompt-kind",
        "raw-resource-uri",
        "non-string-variable-value",
        "non-logical-variable-name",
        "caller-supplied-actor",
    ),
)
def test_mcp_resource_syscalls_reject_nonlogical_inputs_before_facade(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    arguments: dict[str, Any],
    message: str,
) -> None:
    runtime = Runtime.open("local")
    called = False

    async def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("invalid MCP Resource input reached facade")

    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="reject nonlogical MCP Resource input",
        )
        for method_name in (
            "alist_resources",
            "alist_resource_templates",
            "aread_resource",
        ):
            monkeypatch.setattr(
                runtime.mcp,
                method_name,
                unexpected,
                raising=False,
            )

        with pytest.raises(ValidationError, match=message):
            asyncio.run(LibOSSyscallSession(runtime, pid).handle(name, arguments))

        assert called is False
        assert runtime.store.list_external_effects(pid=pid) == []
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        (
            "mcp.resources",
            {"server_id": "registered", "kind": "resource"},
        ),
        (
            "mcp.resource_read",
            {"server_id": "registered", "resource_id": "status"},
        ),
    ],
    ids=("list", "read"),
)
def test_mcp_resource_syscalls_fail_closed_without_protected_facade(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    arguments: dict[str, Any],
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="reject an unprotected MCP Resource call",
        )
        for method_name in (
            "alist_resources",
            "alist_resource_templates",
            "aread_resource",
        ):
            monkeypatch.setattr(runtime.mcp, method_name, None, raising=False)

        with pytest.raises(
            ValidationError,
            match="MCP Resources protected facade is unavailable",
        ):
            asyncio.run(
                LibOSSyscallSession(runtime, pid).handle(name, arguments)
            )

        assert runtime.store.list_external_effects(pid=pid) == []
    finally:
        runtime.close()


def test_mcp_resource_syscalls_route_only_to_model_protected_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    calls: list[tuple[str, dict[str, Any]]] = []

    async def list_resources(
        server_id: str,
        **kwargs: Any,
    ) -> McpPage[McpResource]:
        calls.append(("resources", {"server_id": server_id, **kwargs}))
        return McpPage(items=(McpResource(resource_id="status", name="Status"),))

    async def list_templates(
        server_id: str,
        **kwargs: Any,
    ) -> McpPage[McpResourceTemplate]:
        calls.append(("templates", {"server_id": server_id, **kwargs}))
        return McpPage(
            items=(McpResourceTemplate(template_id="greeting", name="Greeting"),)
        )

    async def read_resource(
        server_id: str,
        resource_id: str,
        **kwargs: Any,
    ) -> McpComplete[McpResourceContents]:
        calls.append(
            (
                "read",
                {
                    "server_id": server_id,
                    "resource_id": resource_id,
                    **kwargs,
                },
            )
        )
        return McpComplete(
            value=McpResourceContents(
                resource_id=resource_id,
                contents=(McpTextContent(text="safe"),),
            )
        )

    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="use protected MCP Resource facade",
        )
        monkeypatch.setattr(
            runtime.mcp,
            "alist_resources",
            list_resources,
            raising=False,
        )
        monkeypatch.setattr(
            runtime.mcp,
            "alist_resource_templates",
            list_templates,
            raising=False,
        )
        monkeypatch.setattr(
            runtime.mcp,
            "aread_resource",
            read_resource,
            raising=False,
        )
        session = LibOSSyscallSession(runtime, pid)

        resources = asyncio.run(
            session.handle(
                "mcp.resources",
                {"server_id": "modern", "kind": "resource"},
            )
        )
        templates = asyncio.run(
            session.handle(
                "mcp.resources",
                {"server_id": "modern", "kind": "template"},
            )
        )
        read = asyncio.run(
            session.handle(
                "mcp.resource_read",
                {
                    "server_id": "modern",
                    "resource_id": "greeting",
                    "variables": {"name": "Ada"},
                },
            )
        )

        assert resources["items"][0]["resource_id"] == "status"
        assert templates["items"][0]["template_id"] == "greeting"
        assert read["result"]["kind"] == "complete"
        assert calls == [
            (
                "resources",
                {
                    "server_id": "modern",
                    "cursor": None,
                    "actor": pid,
                    "model_visible_only": True,
                },
            ),
            (
                "templates",
                {
                    "server_id": "modern",
                    "cursor": None,
                    "actor": pid,
                    "model_visible_only": True,
                },
            ),
            (
                "read",
                {
                    "server_id": "modern",
                    "resource_id": "greeting",
                    "variables": {"name": "Ada"},
                    "actor": pid,
                    "for_model": True,
                },
            ),
        ]
    finally:
        runtime.close()
