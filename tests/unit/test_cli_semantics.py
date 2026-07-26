from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent_libos.api.cli import main as cli_main
from agent_libos.config import DEFAULT_CONFIG


@dataclass(frozen=True)
class _WorkflowResult:
    ok: bool = True


class _FakeProcessManager:
    def __init__(self, captured: list[dict[str, Any] | None]) -> None:
        self._captured = captured

    def spawn(self, **kwargs: Any) -> str:
        self._captured.append(kwargs["authority_manifest"])
        return f"pid-{len(self._captured)}"

    def get(self, _pid: str) -> SimpleNamespace:
        return SimpleNamespace(image_id="base-agent:v0", llm_profile_id="default")


class _FakeRuntime:
    def __init__(self) -> None:
        self.spawn_manifests: list[dict[str, Any] | None] = []
        self.workflow_manifests: list[dict[str, Any] | None] = []
        self.process = _FakeProcessManager(self.spawn_manifests)

    def run_workflow(self, _tool: str, _args: dict[str, Any], **kwargs: Any) -> _WorkflowResult:
        self.workflow_manifests.append(kwargs["authority_manifest"])
        return _WorkflowResult()

    def shutdown(self, **_kwargs: Any) -> dict[str, bool]:
        return {"ok": True}


def test_cli_preserves_absent_and_explicit_empty_authority_manifests(
    monkeypatch,
    capsys,
) -> None:
    runtime = _FakeRuntime()
    monkeypatch.setattr("agent_libos.api.cli.load_config_from_project_root", lambda: DEFAULT_CONFIG)
    monkeypatch.setattr("agent_libos.api.cli.Runtime.open", lambda *_args, **_kwargs: runtime)

    cli_main(["spawn", "--goal", "implicit"])
    cli_main(["spawn", "--goal", "explicit", "--authority-manifest-json", "{}"])
    cli_main(["workflow", "run", "get_working_directory"])
    cli_main(
        [
            "workflow",
            "run",
            "get_working_directory",
            "--authority-manifest-json",
            "{}",
        ]
    )

    capsys.readouterr()
    assert runtime.spawn_manifests == [None, {}]
    assert runtime.workflow_manifests == [None, {}]


def test_modules_verify_does_not_open_a_runtime(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    source = tmp_path / "module.py"
    source_bytes = b"def register_module(context):\n    raise AssertionError('must not execute')\n"
    source.write_bytes(source_bytes)
    manifest = tmp_path / "module.yaml"
    manifest.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "module_id: static-verify:v0",
                "name: Static verify",
                "version: v0",
                "entrypoint: ./module.py:register_module",
                "provides: {}",
                f"sha256: {hashlib.sha256(source_bytes).hexdigest()}",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("agent_libos.api.cli.load_config_from_project_root", lambda: DEFAULT_CONFIG)

    def fail_runtime_open(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("modules verify must not open a Runtime")

    monkeypatch.setattr("agent_libos.api.cli.Runtime.open", fail_runtime_open)

    cli_main(["--db", str(tmp_path / "unused.sqlite"), "modules", "verify", str(manifest)])

    verified = json.loads(capsys.readouterr().out)
    assert verified["module_id"] == "static-verify:v0"
    assert verified["source_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert not (tmp_path / "unused.sqlite").exists()
