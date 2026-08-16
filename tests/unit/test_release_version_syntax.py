from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_release_artifacts import (
    RELEASE_TARGET_VERSION,
    SDIST_EXCLUDE,
    SDIST_INCLUDE,
    _FINAL_RELEASE_VERSION,
    validate_version_alignment,
)


def _write_aligned_release_metadata(root: Path, version: str) -> None:
    (root / "agent_libos").mkdir()
    include = "".join(f'    "{path}",\n' for path in SDIST_INCLUDE)
    exclude = "".join(f'    "{path}",\n' for path in SDIST_EXCLUDE)
    (root / "pyproject.toml").write_text(
        "[project]\n"
        'name = "agent-libos"\n'
        f'version = "{version}"\n\n'
        "[tool.hatch.build.targets.sdist]\n"
        f"include = [\n{include}]\n"
        f"exclude = [\n{exclude}]\n",
        encoding="utf-8",
    )
    (root / "agent_libos" / "__init__.py").write_text(
        f'__version__ = "{version}"\n',
        encoding="utf-8",
    )
    (root / "uv.lock").write_text(
        "version = 1\n\n"
        "[[package]]\n"
        'name = "agent-libos"\n'
        f'version = "{version}"\n'
        'source = { editable = "." }\n',
        encoding="utf-8",
    )
    (root / "experiments" / "agentdojo").mkdir(parents=True)
    (root / "experiments" / "agentdojo" / "uv.lock").write_text(
        "version = 1\n\n"
        "[[package]]\n"
        'name = "agent-libos"\n'
        f'version = "{version}"\n'
        'source = { editable = "../../" }\n',
        encoding="utf-8",
    )
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / ".github" / "workflows" / "test.yml").write_text(
        "env:\n"
        f"  RELEASE_WHEEL: dist/agent_libos-{version}-py3-none-any.whl\n"
        f"  RELEASE_SDIST: dist/agent_libos-{version}.tar.gz\n",
        encoding="utf-8",
    )
    (root / ".github" / "workflows" / "desktop-internal.yml").write_text(
        "env:\n"
        f'  AGENT_LIBOS_DESKTOP_VERSION: "{version}"\n'
        "jobs:\n"
        "  package:\n"
        "    strategy:\n"
        "      matrix:\n"
        "        include:\n"
        f"          - upload_name: agent-libos-{version}-macos-arm64-internal-unsigned\n"
        f"          - upload_name: agent-libos-{version}-windows-x64-internal-unsigned\n"
        f"          - upload_name: agent-libos-{version}-linux-x64-internal-unsigned\n"
        f"    pattern: agent-libos-{version}-*-internal-unsigned-${{{{ github.sha }}}}-${{{{ github.run_attempt }}}}\n",
        encoding="utf-8",
    )
    (root / "skills" / "swe-agent").mkdir(parents=True)
    (root / "skills" / "swe-agent" / "SKILL.md").write_text(
        f"compatibility: agent-libos=={version}\n",
        encoding="utf-8",
    )
    (root / "agent_libos" / "substrate").mkdir()
    (root / "agent_libos" / "substrate" / "local.py").write_text(
        "def _mcp_client_info(governed_modern):\n"
        f'    return dict(version="{version}" if governed_modern else "1.4.2")\n',
        encoding="utf-8",
    )
    (root / "desktop").mkdir()
    (root / "desktop" / "runtime-manifest.json").write_text(
        f'{{"product": {{"version": "{version}"}}}}\n',
        encoding="utf-8",
    )
    (root / "scripts").mkdir()
    for relative_path in (
        "check_desktop_artifacts.py",
        "verify_desktop_installers.py",
    ):
        (root / "scripts" / relative_path).write_text(
            f'VERSION = "{version}"\n',
            encoding="utf-8",
        )
    (root / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## Unreleased\n\n## {version}\n",
        encoding="utf-8",
    )
    (root / "docs").mkdir()
    (root / "docs" / "release_status.md").write_text(
        f"# Agent libOS {version} Status\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("version", ["0.0.0", "2.0.0", "10.20.30"])
def test_release_version_syntax_accepts_final_numeric_ascii_triplet(
    version: str,
) -> None:
    assert _FINAL_RELEASE_VERSION.fullmatch(version) is not None


def test_release_version_accepts_only_the_current_target(tmp_path: Path) -> None:
    _write_aligned_release_metadata(tmp_path, RELEASE_TARGET_VERSION)

    assert validate_version_alignment(tmp_path) == "1.5.1"


@pytest.mark.parametrize("version", ["0.0.0", "1.4.2", "2.0.0", "10.20.30"])
def test_release_version_rejects_aligned_non_target_version(
    tmp_path: Path,
    version: str,
) -> None:
    _write_aligned_release_metadata(tmp_path, version)

    with pytest.raises(ValueError, match="target version must be exactly 1.5.1"):
        validate_version_alignment(tmp_path)


@pytest.mark.parametrize(
    "version",
    [
        "02.0.0",
        "2.00.0",
        "2.0.00",
        "2.0",
        "2.0.0.0",
        "2.0.0rc1",
        "2.0.0+local",
        "v2.0.0",
        "２.0.0",
    ],
)
def test_release_version_rejects_non_final_or_non_ascii_spelling(
    tmp_path: Path,
    version: str,
) -> None:
    _write_aligned_release_metadata(tmp_path, version)

    with pytest.raises(ValueError, match="final-form numeric X.Y.Z"):
        validate_version_alignment(tmp_path)


@pytest.mark.parametrize(
    ("relative_path", "old", "new", "source"),
    [
        (
            "experiments/agentdojo/uv.lock",
            'version = "1.5.1"',
            'version = "1.5.2"',
            "experiments/agentdojo/uv.lock",
        ),
        (
            ".github/workflows/test.yml",
            "agent_libos-1.5.1-py3-none-any.whl",
            "agent_libos-1.5.2-py3-none-any.whl",
            "RELEASE_WHEEL",
        ),
        (
            ".github/workflows/test.yml",
            "agent_libos-1.5.1.tar.gz",
            "agent_libos-1.5.2.tar.gz",
            "RELEASE_SDIST",
        ),
        (
            "skills/swe-agent/SKILL.md",
            "agent-libos==1.5.1",
            "agent-libos==1.5.2",
            "skills/swe-agent/SKILL.md",
        ),
        (
            "agent_libos/substrate/local.py",
            'version="1.5.1"',
            'version="1.5.2"',
            "MCP clientInfo",
        ),
        (
            "desktop/runtime-manifest.json",
            '"version": "1.5.1"',
            '"version": "1.5.2"',
            "desktop/runtime-manifest.json",
        ),
        (
            "scripts/check_desktop_artifacts.py",
            'VERSION = "1.5.1"',
            'VERSION = "1.5.2"',
            "scripts/check_desktop_artifacts.py",
        ),
        (
            "CHANGELOG.md",
            "## 1.5.1",
            "## 1.5.2",
            "CHANGELOG.md current release",
        ),
        (
            "docs/release_status.md",
            "Agent libOS 1.5.1 Status",
            "Agent libOS 1.5.2 Status",
            "docs/release_status.md",
        ),
    ],
)
def test_release_version_detects_extended_surface_mismatch(
    tmp_path: Path,
    relative_path: str,
    old: str,
    new: str,
    source: str,
) -> None:
    _write_aligned_release_metadata(tmp_path, RELEASE_TARGET_VERSION)
    path = tmp_path / relative_path
    path.write_text(
        path.read_text(encoding="utf-8").replace(old, new),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=source):
        validate_version_alignment(tmp_path)


def test_release_validation_rejects_sdist_partition_drift(tmp_path: Path) -> None:
    _write_aligned_release_metadata(tmp_path, RELEASE_TARGET_VERSION)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            '    "/plan.md",\n',
            '    "/ordinary-unclassified.txt",\n',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exact explicit include/exclude partition"):
        validate_version_alignment(tmp_path)


def test_release_validation_rejects_desktop_artifact_name_drift(
    tmp_path: Path,
) -> None:
    _write_aligned_release_metadata(tmp_path, RELEASE_TARGET_VERSION)
    workflow = tmp_path / ".github" / "workflows" / "desktop-internal.yml"
    workflow.write_text(
        workflow.read_text(encoding="utf-8").replace(
            "agent-libos-1.5.1-macos-arm64-internal-unsigned",
            "agent-libos-1.5.2-macos-arm64-internal-unsigned",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="desktop workflow artifact names"):
        validate_version_alignment(tmp_path)


def test_release_runbook_keeps_remote_version_uniqueness_out_of_local_checker() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "docs" / "releasing.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert (
        "checker proves only that checked-in version identifiers are aligned"
        in normalized
    )
    assert (
        "does not contact a package index or inspect remote Git references"
        in normalized
    )
    assert "re-check the complete PyPI project history" in normalized
    assert "intended remote's tags" in normalized
    assert "pins the exact target `1.5.1`" in normalized


def test_release_runbook_builds_local_preflight_in_a_fresh_directory() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "docs" / "releasing.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert 'LOCAL_RELEASE_DIR="$(mktemp -d)"' in text
    assert (
        'LOCAL_RELEASE_WHEEL="$LOCAL_RELEASE_DIR/'
        'agent_libos-${RELEASE_VERSION}-py3-none-any.whl"'
    ) in text
    assert (
        'LOCAL_RELEASE_SDIST="$LOCAL_RELEASE_DIR/'
        'agent_libos-${RELEASE_VERSION}.tar.gz"'
    ) in text
    assert '--out-dir "$LOCAL_RELEASE_DIR"' in normalized
    assert text.count(
        'scripts/check_release_artifacts.py "$LOCAL_RELEASE_DIR"'
    ) == 2
    assert "never clears or writes the repository's existing `dist/`" in normalized
    assert "--clear --out-dir dist" not in text
