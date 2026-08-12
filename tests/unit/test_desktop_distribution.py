from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from scripts import build_desktop_backend
from scripts import check_desktop_artifacts as checker
from scripts import finalize_desktop_artifacts as finalizer
from scripts import stage_desktop_runtime


ROOT = Path(__file__).resolve().parents[2]


def test_desktop_runtime_manifest_locks_exact_internal_toolchain() -> None:
    manifest = json.loads(
        (ROOT / "desktop" / "runtime-manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["schema_version"] == 1
    assert manifest["distribution_channel"] == "internal-unsigned"
    assert manifest["product"] == {
        "name": "Agent libOS",
        "version": "1.5.0",
        "app_id": "io.agentlibos.desktop",
    }
    assert manifest["toolchain"] == {
        "python": "3.11.15",
        "pyinstaller": "6.21.0",
        "node": "24.15.0",
        "electron": "43.2.0",
        "electron_builder": "26.15.3",
        "deno": "2.9.4",
        "mcp": "2.0.0",
        "keyring": "25.7.0",
    }
    assert manifest["deno"]["targets"] == {
        "darwin-arm64": {
            "asset": "deno-aarch64-apple-darwin.zip",
            "sha256": "6d17647fdbf9c587a581dba205054c4ccf732dae0a196cc1e9b44c07589db412",
        },
        "win32-x64": {
            "asset": "deno-x86_64-pc-windows-msvc.zip",
            "sha256": "68ed08b05c56cf887e9aa509947dc3f468f7e12f47a13e5c1abd51d46d1453ef",
        },
        "linux-x64": {
            "asset": "deno-x86_64-unknown-linux-gnu.zip",
            "sha256": "c24f955d9fbfe0ea5ae2b501c8e71ae76e31e4c9782390a54a284b3364fda725",
        },
    }


def test_desktop_build_metadata_and_scripts_use_exact_versions_and_no_publish() -> None:
    package = json.loads((ROOT / "gui" / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "gui" / "package-lock.json").read_text(encoding="utf-8"))
    config = (ROOT / "gui" / "electron-builder.yml").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert package["version"] == "1.5.0"
    assert package["devDependencies"]["electron"] == "43.2.0"
    assert package["devDependencies"]["electron-builder"] == "26.15.3"
    assert lock["packages"]["node_modules/electron"]["version"] == "43.2.0"
    assert lock["packages"]["node_modules/electron-builder"]["version"] == "26.15.3"
    assert "pyinstaller==6.21.0" in pyproject
    assert "appId: io.agentlibos.desktop" in config
    assert "identity: '-'" in config
    assert "publish: null" in config
    assert "--publish never" in package["scripts"]["desktop:dist"]
    assert "check_desktop_artifacts.py" in package["scripts"]["desktop:dist"]


def test_internal_desktop_workflow_is_manual_native_and_artifact_only() -> None:
    workflow = (ROOT / ".github" / "workflows" / "desktop-internal.yml").read_text(
        encoding="utf-8"
    )

    assert "workflow_dispatch:" in workflow
    for runner in ("macos-15", "windows-2025", "ubuntu-24.04"):
        assert f"runner: {runner}" in workflow
    for target in ("darwin-arm64", "win32-x64", "linux-x64"):
        assert f"target: {target}" in workflow
    assert "actions/upload-artifact@" in workflow
    assert "contents: read" in workflow
    assert "desktop:dist" in workflow
    assert "smoke_desktop_bundle.py" in workflow
    assert "verify_desktop_installers.py" in workflow
    assert "action-gh-release" not in workflow
    assert "gh release" not in workflow


def test_desktop_release_names_are_exact_for_three_native_targets() -> None:
    assert finalizer._package_names("1.5.0", "darwin-arm64") == (
        "Agent-libOS-1.5.0-macos-arm64.dmg",
        "Agent-libOS-1.5.0-macos-arm64.zip",
    )
    assert finalizer._package_names("1.5.0", "win32-x64") == (
        "Agent-libOS-Setup-1.5.0-windows-x64.exe",
        "Agent-libOS-1.5.0-windows-x64.zip",
    )
    assert finalizer._package_names("1.5.0", "linux-x64") == (
        "Agent-libOS-1.5.0-linux-x64.AppImage",
        "Agent-libOS-1.5.0-linux-x64.tar.gz",
    )


def test_desktop_helpers_reject_broad_output_and_unsafe_archive_paths(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="broad"):
        build_desktop_backend._safe_output_path(ROOT)
    with pytest.raises(RuntimeError, match="broad"):
        stage_desktop_runtime._safe_stage_path(ROOT / "gui")
    with pytest.raises(ValueError, match="unsafe path"):
        checker._safe_archive_path("../secret.env")
    with pytest.raises(ValueError, match="mutable/private"):
        checker._safe_archive_path("Agent/resources/operator.sqlite")
    checker._safe_archive_path(
        "Agent/resources/backend/_internal/certifi/cacert.pem"
    )
    with pytest.raises(ValueError, match="mutable/private"):
        checker._safe_archive_path("Agent/resources/backend/client-secret.pem")

    package = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("../escape", "bad")
    with pytest.raises(ValueError, match="unsafe path"):
        checker._validate_zip(package)


def test_editable_checkout_receipt_is_removed_from_frozen_metadata(tmp_path: Path) -> None:
    metadata = tmp_path / "_internal" / "agent_libos-1.5.0.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "direct_url.json").write_text(
        '{"url":"file:///private/checkout","dir_info":{"editable":true}}',
        encoding="utf-8",
    )
    (metadata / "RECORD").write_text(
        "agent_libos-1.5.0.dist-info/direct_url.json,sha256=abc,1\n"
        "agent_libos-1.5.0.dist-info/METADATA,sha256=def,2\n",
        encoding="utf-8",
    )

    build_desktop_backend._remove_editable_install_receipt(tmp_path)

    assert not (metadata / "direct_url.json").exists()
    assert "direct_url.json" not in (metadata / "RECORD").read_text(encoding="utf-8")


def test_desktop_finalizer_removes_unpublished_update_and_debug_metadata(
    tmp_path: Path,
) -> None:
    for name in (
        "Agent-libOS-1.5.0-macos-arm64.zip.blockmap",
        "Agent-libOS-1.5.0-macos-arm64.dmg.blockmap",
        "builder-debug.yml",
        "builder-effective-config.yaml",
    ):
        (tmp_path / name).write_text("not a release artifact", encoding="utf-8")
    icon_cache = tmp_path / ".icon-icns"
    icon_cache.mkdir()
    (icon_cache / "icon.icns").write_bytes(b"generated")

    finalizer._remove_unpublished_builder_metadata(tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_desktop_sbom_requires_runtime_mcp_keyring_electron_deno_and_python(
    tmp_path: Path,
) -> None:
    components = [
        {"name": name, "version": version}
        for name, version in {
            "agent-libos": "1.5.0",
            "CPython": "3.11.15",
            "Deno": "2.9.4",
            "electron": "43.2.0",
            "keyring": "25.7.0",
            "mcp": "2.0.0",
        }.items()
    ]
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "metadata": {
            "component": {
                "name": "Agent libOS",
                "version": "1.5.0",
                "properties": [
                    {
                        "name": "agent-libos:distribution-channel",
                        "value": "internal-unsigned",
                    },
                    {"name": "agent-libos:target", "value": "darwin-arm64"},
                ],
            }
        },
        "components": components,
    }
    path = tmp_path / "sbom.json"
    path.write_text(json.dumps(sbom), encoding="utf-8")

    checker._validate_sbom(path, "darwin-arm64")
    components.pop()
    path.write_text(json.dumps(sbom), encoding="utf-8")
    with pytest.raises(ValueError, match="exact component versions"):
        checker._validate_sbom(path, "darwin-arm64")
