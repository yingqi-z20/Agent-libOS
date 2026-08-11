from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.images import DEFAULT_IMAGES
from agent_libos.models import (
    AgentImage,
    CapabilityRight,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    ProcessStatus,
)
from agent_libos.skills import get_builtin_skill_catalog
from agent_libos.substrate import CommandResult, LocalResourceProviderSubstrate
from tests.support.fakes import RecordingActionClient


_SKILL_BOOTSTRAP = [
    "discover_skills",
    "activate_skill",
    "read_skill_resource",
    "unload_skill",
    "process_exit",
]


def _activate_action(skill_id: str) -> dict[str, str]:
    package = get_builtin_skill_catalog().get(skill_id)
    assert package is not None
    return {
        "action": "activate_skill",
        "skill_id": skill_id,
        "expected_package_sha256": package.package_sha256,
    }


def _one_phase_coding_image(
    image_id: str,
    *,
    default_skills: list[str] | None = None,
) -> AgentImage:
    """Isolate Tool-Skill routing from the separately tested exit review."""

    base = DEFAULT_IMAGES["coding-agent:v0"]
    metadata = dict(base.metadata)
    metadata.pop("completion_gate", None)
    return replace(
        base,
        image_id=image_id,
        name=image_id.removesuffix(":v0"),
        default_skills=(
            list(base.default_skills)
            if default_skills is None
            else list(default_skills)
        ),
        metadata=metadata,
    )


def test_agent_routes_file_editing_through_builtin_skill_and_verifies_content(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(
        ":memory:",
        substrate=LocalResourceProviderSubstrate(tmp_path),
    )
    image = _one_phase_coding_image("file-editing-routing-agent:v0")
    runtime.register_image(image, actor="cli")
    client = RecordingActionClient(
        [
            {
                "action": "discover_skills",
                "text": "agent-libos-workspace-editing",
            },
            _activate_action("agent-libos-workspace-editing"),
            {
                "action": "write_text_file",
                "path": "routed.txt",
                "content": "skill-routed\n",
            },
            {
                "action": "discover_skills",
                "text": "agent-libos-workspace-navigation",
            },
            _activate_action("agent-libos-workspace-navigation"),
            {"action": "read_text_file", "path": "routed.txt"},
            {"action": "process_exit", "payload": {"verified": True}},
        ]
    )
    runtime.llm.client = client
    try:
        pid = runtime.process.spawn(
            image=image.image_id,
            goal="Create routed.txt through the applicable built-in Tool Skill and verify it.",
        )
        runtime.filesystem.grant_path(
            pid,
            "routed.txt",
            [CapabilityRight.READ, CapabilityRight.WRITE],
            issued_by="builtin-skill-routing-test",
        )
        _assert_initially_hidden(runtime, pid, "write_text_file")

        results = runtime.run_process_until_idle(pid, max_quanta=8)

        _assert_standard_route(
            runtime,
            pid,
            client,
            results,
            skill_id="agent-libos-workspace-editing",
            projected_tool="write_text_file",
            expected_actions=[
                "discover_skills",
                "activate_skill",
                "write_text_file",
                "discover_skills",
                "activate_skill",
                "read_text_file",
                "process_exit",
            ],
        )
        assert _payload(results, "read_text_file")["content"] == "skill-routed\n"
        assert tmp_path.joinpath("routed.txt").read_text(encoding="utf-8") == "skill-routed\n"
        assert "# Edit the workspace" not in client.user_prompts[0]
        assert "# Edit the workspace" not in client.user_prompts[1]
        assert "# Edit the workspace" in client.user_prompts[2]
    finally:
        runtime.close()


def test_agent_routes_git_inspection_through_builtin_skill_and_verifies_state(
    tmp_path: Path,
) -> None:
    _init_git_repository(tmp_path)
    runtime = Runtime.open(
        ":memory:",
        substrate=LocalResourceProviderSubstrate(tmp_path),
    )
    client = RecordingActionClient(
        [
            {
                "action": "discover_skills",
                "text": "agent-libos-git-inspection",
            },
            _activate_action("agent-libos-git-inspection"),
            {"action": "git_status"},
            {"action": "git_repository_info"},
            {"action": "process_exit", "payload": {"verified": True}},
        ]
    )
    runtime.llm.client = client
    try:
        pid = runtime.process.spawn(
            image="review-agent:v0",
            goal="Inspect and verify the current Git repository state.",
        )
        runtime.capability.issue_trusted(
            pid,
            "git:workspace",
            [CapabilityRight.READ, CapabilityRight.DIFF],
            issued_by="builtin-skill-routing-test",
        )
        _assert_initially_hidden(runtime, pid, "git_status")

        results = runtime.run_process_until_idle(pid, max_quanta=8)

        _assert_standard_route(
            runtime,
            pid,
            client,
            results,
            skill_id="agent-libos-git-inspection",
            projected_tool="git_status",
            expected_actions=[
                "discover_skills",
                "activate_skill",
                "git_status",
                "git_repository_info",
                "process_exit",
            ],
        )
        status = _payload(results, "git_status")
        repository = _payload(results, "git_repository_info")
        assert status["branch"] == "main"
        assert status["entries"] == []
        assert repository["state"]["token"] == status["state"]["token"]
    finally:
        runtime.close()


def test_agent_routes_shell_execution_through_builtin_skill_and_verifies_output(
    tmp_path: Path,
) -> None:
    shell = _RecordingShellProvider()
    substrate = LocalResourceProviderSubstrate(tmp_path)
    substrate.shell = shell
    runtime = Runtime.open(":memory:", substrate=substrate)
    image = _one_phase_coding_image("shell-routing-agent:v0")
    runtime.register_image(image, actor="cli")
    client = RecordingActionClient(
        [
            {
                "action": "discover_skills",
                "text": "agent-libos-command-execution",
            },
            _activate_action("agent-libos-command-execution"),
            {
                "action": "run_shell_command",
                "argv": ["routing-check", "--inspect"],
            },
            {
                "action": "run_shell_command",
                "argv": ["routing-check", "--verify"],
            },
            {"action": "process_exit", "payload": {"verified": True}},
        ]
    )
    runtime.llm.client = client
    try:
        pid = runtime.process.spawn(
            image=image.image_id,
            goal="Run a governed command and verify its bounded output.",
        )
        runtime.shell.grant_policy(
            pid,
            runtime.config.shell.always_allow_level,
            issued_by="builtin-skill-routing-test",
        )
        _assert_initially_hidden(runtime, pid, "run_shell_command")

        results = runtime.run_process_until_idle(pid, max_quanta=8)

        _assert_standard_route(
            runtime,
            pid,
            client,
            results,
            skill_id="agent-libos-command-execution",
            projected_tool="run_shell_command",
            expected_actions=[
                "discover_skills",
                "activate_skill",
                "run_shell_command",
                "run_shell_command",
                "process_exit",
            ],
        )
        assert _payload(results, "run_shell_command", occurrence=0)["stdout"] == "routing-ok\n"
        assert _payload(results, "run_shell_command", occurrence=1)["stdout"] == "routing-ok\n"
        assert shell.calls == [
            (["routing-check", "--inspect"], "."),
            (["routing-check", "--verify"], "."),
        ]
    finally:
        runtime.close()


def test_agent_routes_checkpoint_inspection_through_builtin_skill(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(
        ":memory:",
        substrate=LocalResourceProviderSubstrate(tmp_path),
        config=replace(
            DEFAULT_CONFIG,
            llm=replace(
                DEFAULT_CONFIG.llm,
                prompt_layout="cache_optimized_v2",
            ),
        ),
    )
    catalog = get_builtin_skill_catalog()
    checkpoint_skill = catalog.get("agent-libos-checkpoints")
    assert checkpoint_skill is not None
    image = AgentImage(
        image_id="checkpoint-routing-agent:v0",
        name="checkpoint-routing-agent",
        system_prompt="Inspect checkpoints and verify the selected snapshot.",
        default_tools=list(
            dict.fromkeys(
                [
                    *_SKILL_BOOTSTRAP,
                    *checkpoint_skill.allowed_tools,
                ]
            )
        ),
        default_skills=[],
        metadata={"tool_projection": "skills"},
    )
    runtime.register_image(image, actor="cli")
    try:
        pid = runtime.process.spawn(
            image=image.image_id,
            goal="Discover checkpoint tools, inspect the saved state, and exit.",
        )
        checkpoint_id = runtime.checkpoint.create(
            pid,
            "deterministic routing fixture",
            actor=pid,
        )
        client = RecordingActionClient(
            [
                {
                    "action": "discover_skills",
                    "text": "agent-libos-checkpoints",
                },
                    _activate_action("agent-libos-checkpoints"),
                {"action": "list_checkpoints"},
                {
                    "action": "inspect_checkpoint",
                    "checkpoint_id": "only",
                },
                {"action": "process_exit", "payload": {"verified": True}},
            ]
        )
        runtime.llm.client = client
        _assert_initially_hidden(runtime, pid, "list_checkpoints")

        results = runtime.run_process_until_idle(pid, max_quanta=8)

        _assert_standard_route(
            runtime,
            pid,
            client,
            results,
            skill_id="agent-libos-checkpoints",
            projected_tool="list_checkpoints",
            expected_actions=[
                "discover_skills",
                "activate_skill",
                "list_checkpoints",
                "inspect_checkpoint",
                "process_exit",
            ],
        )
        listed = _payload(results, "list_checkpoints")["checkpoints"]
        inspected = _payload(results, "inspect_checkpoint")
        assert listed == [
            {
                "checkpoint_ref": "only",
                "reason": "deterministic routing fixture",
            }
        ]
        assert checkpoint_id not in json.dumps(listed, sort_keys=True)
        assert inspected["checkpoint"]["checkpoint_id"] == checkpoint_id
        assert inspected["subtree_pids"] == [pid]
    finally:
        runtime.close()


def test_agent_routes_mcp_registry_reads_through_builtin_skill(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(
        ":memory:",
        substrate=LocalResourceProviderSubstrate(tmp_path),
    )
    runtime.mcp.register_server_from_yaml_text(
        _mcp_manifest("routing-mcp"),
        actor="cli",
        require_capability=False,
    )
    client = RecordingActionClient(
        [
            {"action": "discover_skills", "text": "agent-libos-mcp"},
                _activate_action("agent-libos-mcp"),
            {"action": "list_mcp_servers"},
            {"action": "inspect_mcp_server", "server_id": "routing-mcp"},
            {"action": "process_exit", "payload": {"verified": True}},
        ]
    )
    runtime.llm.client = client
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="Discover the registered MCP server, inspect it, and exit.",
        )
        runtime.capability.grant(
            pid,
            runtime.config.mcp.registry_resource,
            [CapabilityRight.READ],
            issued_by="builtin-skill-routing-test",
        )
        runtime.capability.grant(
            pid,
            runtime.mcp.server_resource("routing-mcp"),
            [CapabilityRight.READ],
            issued_by="builtin-skill-routing-test",
        )
        _assert_initially_hidden(runtime, pid, "list_mcp_servers")

        results = runtime.run_process_until_idle(pid, max_quanta=8)

        _assert_standard_route(
            runtime,
            pid,
            client,
            results,
            skill_id="agent-libos-mcp",
            projected_tool="list_mcp_servers",
            expected_actions=[
                "discover_skills",
                "activate_skill",
                "list_mcp_servers",
                "inspect_mcp_server",
                "process_exit",
            ],
        )
        servers = _payload(results, "list_mcp_servers")["servers"]
        inspected = _payload(results, "inspect_mcp_server")["server"]
        assert [item["server_id"] for item in servers] == ["routing-mcp"]
        assert inspected["server_id"] == "routing-mcp"
        assert [item["tool_id"] for item in inspected["tools"]] == ["echo"]
    finally:
        runtime.close()


def test_adjacent_navigation_skill_near_miss_does_not_project_editing_tools(
    tmp_path: Path,
) -> None:
    config = replace(
        DEFAULT_CONFIG,
        llm=replace(DEFAULT_CONFIG.llm, action_repair_attempts=2),
    )
    runtime = Runtime.open(
        ":memory:",
        config=config,
        substrate=LocalResourceProviderSubstrate(tmp_path),
    )
    image = _one_phase_coding_image(
        "adjacent-skill-routing-agent:v0",
        default_skills=[],
    )
    runtime.register_image(image, actor="cli")
    client = RecordingActionClient(
        [
            {"action": "discover_skills", "text": "agent-libos-workspace"},
                _activate_action("agent-libos-workspace-navigation"),
            {
                "action": "write_text_file",
                "path": "near-miss.txt",
                "content": "recovered\n",
            },
                _activate_action("agent-libos-workspace-editing"),
            {
                "action": "write_text_file",
                "path": "near-miss.txt",
                "content": "recovered\n",
            },
            {"action": "read_text_file", "path": "near-miss.txt"},
            {"action": "process_exit", "payload": {"verified": True}},
        ]
    )
    runtime.llm.client = client
    try:
        pid = runtime.process.spawn(
            image=image.image_id,
            goal=(
                "Navigate the workspace, then create near-miss.txt using the "
                "separate editing Skill and verify it."
            ),
        )
        runtime.filesystem.grant_path(
            pid,
            "near-miss.txt",
            [CapabilityRight.READ, CapabilityRight.WRITE],
            issued_by="builtin-skill-routing-test",
        )
        _assert_initially_hidden(runtime, pid, "read_text_file")
        _assert_initially_hidden(runtime, pid, "write_text_file")

        results = runtime.run_process_until_idle(pid, max_quanta=9)

        assert _action_names(results) == [
            "discover_skills",
            "activate_skill",
            "activate_skill",
            "write_text_file",
            "read_text_file",
            "process_exit",
        ]
        assert runtime.process.get(pid).status == ProcessStatus.EXITED
        assert client.actions == []
        batches = [_tool_names(batch) for batch in client.tool_batches]
        assert "read_text_file" not in batches[0]
        assert "write_text_file" not in batches[0]
        assert "read_text_file" not in batches[1]
        assert "write_text_file" not in batches[1]
        assert "read_text_file" in batches[2]
        assert "write_text_file" not in batches[2]
        assert "read_text_file" in batches[3]
        assert "write_text_file" not in batches[3]
        assert "write_text_file" in batches[4]
        assert _payload(results, "read_text_file")["content"] == "recovered\n"
        assert tmp_path.joinpath("near-miss.txt").read_text(encoding="utf-8") == "recovered\n"

        repairs = [
            record
            for record in runtime.audit.trace()
            if record.action == "llm.action_repair_requested"
        ]
        assert len(repairs) == 1
        assert (
            repairs[0].decision["error"]
            == "selected action is not in this process model tool projection: write_text_file"
        )
        assert repairs[0].decision["tool_calls_preview"][0]["name"] == "write_text_file"
    finally:
        runtime.close()


def _assert_initially_hidden(
    runtime: Runtime,
    pid: str,
    tool_name: str,
) -> None:
    process = runtime.process.get(pid)
    assert tool_name in process.tool_table
    assert tool_name not in process.model_tool_table


def _assert_standard_route(
    runtime: Runtime,
    pid: str,
    client: RecordingActionClient,
    results: list[Any],
    *,
    skill_id: str,
    projected_tool: str,
    expected_actions: list[str],
) -> None:
    assert runtime.process.get(pid).status == ProcessStatus.EXITED
    assert _action_names(results) == expected_actions
    assert client.actions == []
    assert all(item["result"]["ok"] for item in results)
    assert all(
        marker not in prompt
        for prompt in client.user_prompts
        for marker in (
            "builtin_projection",
            '"source_type":"builtin"',
            '"activation_kind"',
        )
    )

    discovered = _payload(results, "discover_skills")
    matching = [item for item in discovered["skills"] if item["skill_id"] == skill_id]
    assert len(matching) == 1
    assert matching[0]["active"] is False
    assert discovered["visibility_limited"] is True
    assert {
        "source",
        "source_type",
        "catalog_scope",
        "available_tools",
    }.isdisjoint(matching[0])

    activation = _payload(results, "activate_skill")["result"]
    assert activation["skill_id"] == skill_id
    assert {
        "activation_kind",
        "authority_changed",
        "source",
        "source_type",
        "registered",
    }.isdisjoint(activation)

    batches = [_tool_names(batch) for batch in client.tool_batches]
    assert projected_tool not in batches[0]
    assert projected_tool not in batches[1]
    assert projected_tool in batches[2]


def _action_names(results: list[Any]) -> list[str]:
    return [
        str(item["action"]["action"])
        for item in results
        if isinstance(item, dict)
        and isinstance(item.get("action"), dict)
        and isinstance(item["action"].get("action"), str)
    ]


def _payload(
    results: list[Any],
    action_name: str,
    *,
    occurrence: int = 0,
) -> dict[str, Any]:
    matches = [
        item["result"]
        for item in results
        if isinstance(item, dict)
        and isinstance(item.get("action"), dict)
        and item["action"].get("action") == action_name
    ]
    assert len(matches) > occurrence
    selected = matches[occurrence]
    assert selected["ok"], selected
    assert isinstance(selected["payload"], dict)
    return selected["payload"]


def _tool_names(batch: list[dict[str, Any]]) -> set[str]:
    return {str(tool["function"]["name"]) for tool in batch}


def _init_git_repository(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "symbolic-ref", "HEAD", "refs/heads/main")
    _git(root, "config", "user.name", "Agent libOS Test")
    _git(root, "config", "user.email", "agent-libos@example.test")
    root.joinpath("tracked.txt").write_text("initial\n", encoding="utf-8")
    _git(root, "add", "--", "tracked.txt")
    _git(root, "commit", "-q", "-m", "initial")


def _git(root: Path, *args: str) -> None:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8",
        errors="replace",
    )


class _RecordingShellProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str | None]] = []

    def run(
        self,
        argv: list[str],
        *,
        timeout: float = 30.0,
        cwd: str | None = None,
        limits: object | None = None,
        stdout_limit_chars: int | None = None,
        stderr_limit_chars: int | None = None,
    ) -> CommandResult:
        del timeout, limits, stdout_limit_chars, stderr_limit_chars
        self.calls.append((list(argv), cwd))
        return CommandResult(
            argv=list(argv),
            returncode=0,
            stdout="routing-ok\n",
            stderr="",
        )

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification:
        del context, result
        return ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
            rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
            state_mutation=False,
            information_flow=True,
            metadata={"operation": operation},
        )


def _mcp_manifest(server_id: str) -> str:
    return f"""
schema_version: 1
server_id: {server_id}
transport: stdio
stdio:
  command: {json.dumps(sys.executable)}
  args: ["-m", "demo_server"]
tools:
  - tool_id: echo
    mcp_name: demo.echo
    right: read
    rollback_class: no_rollback_required
    state_mutation: false
    information_flow: true
    input_schema:
      type: object
      additionalProperties: false
timeout_s: 5
max_request_bytes: 65536
max_response_bytes: 1048576
""".strip()
