from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
GUI_ROOT = ROOT / "gui"
MANIFEST_PATH = ROOT / "desktop" / "runtime-manifest.json"
SPEC_PATH = ROOT / "desktop" / "pyinstaller" / "agent_libos_gui_server.spec"


def _manifest() -> dict[str, object]:
    value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("desktop runtime manifest must be an object")
    return value


def _require_build_versions(manifest: dict[str, object]) -> None:
    toolchain = manifest.get("toolchain")
    product = manifest.get("product")
    if not isinstance(toolchain, dict) or not isinstance(product, dict):
        raise RuntimeError("desktop runtime manifest is malformed")
    expected_python = str(toolchain["python"])
    selected_python = ".".join(str(part) for part in sys.version_info[:3])
    if selected_python != expected_python:
        raise RuntimeError(
            f"desktop backend requires Python {expected_python}, got {selected_python}"
        )
    expected = {
        "agent-libos": str(product["version"]),
        "keyring": str(toolchain["keyring"]),
        "mcp": str(toolchain["mcp"]),
        "pyinstaller": str(toolchain["pyinstaller"]),
    }
    for distribution, version in expected.items():
        selected = importlib.metadata.version(distribution)
        if selected != version:
            raise RuntimeError(
                f"desktop backend requires {distribution}=={version}, got {selected}"
            )


def _safe_output_path(value: str | Path) -> Path:
    selected = Path(value).expanduser().resolve()
    if selected in {Path(selected.anchor), ROOT, GUI_ROOT}:
        raise RuntimeError("refusing broad desktop backend output path")
    return selected


def _remove_editable_install_receipt(built: Path) -> None:
    """Remove the local-checkout URL emitted by an editable build environment."""

    matches = list((built / "_internal").glob("agent_libos-*.dist-info"))
    if len(matches) != 1:
        raise RuntimeError("frozen backend must contain one Agent libOS distribution record")
    metadata_root = matches[0]
    direct_url = metadata_root / "direct_url.json"
    if direct_url.exists():
        if direct_url.is_symlink() or not direct_url.is_file():
            raise RuntimeError("frozen backend direct_url metadata is invalid")
        direct_url.unlink()
    record = metadata_root / "RECORD"
    if record.is_file():
        kept = [
            line
            for line in record.read_text(encoding="utf-8").splitlines()
            if not line.startswith(f"{metadata_root.name}/direct_url.json,")
        ]
        record.write_text("\n".join(kept) + "\n", encoding="utf-8")


def build_backend(output_dir: Path, work_dir: Path) -> Path:
    manifest = _manifest()
    _require_build_versions(manifest)
    output_dir = _safe_output_path(output_dir)
    work_dir = _safe_output_path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix="backend-", dir=work_dir))
    try:
        dist_path = temporary_root / "dist"
        build_path = temporary_root / "work"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "PyInstaller",
                "--clean",
                "--noconfirm",
                "--distpath",
                str(dist_path),
                "--workpath",
                str(build_path),
                str(SPEC_PATH),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "PYINSTALLER_CONFIG_DIR": str(temporary_root / "config"),
            },
            check=True,
        )
        built = dist_path / "agent-libos-gui-server"
        executable = built / (
            "agent-libos-gui-server.exe" if os.name == "nt" else "agent-libos-gui-server"
        )
        if not executable.is_file():
            raise RuntimeError("PyInstaller did not produce the GUI server executable")
        _remove_editable_install_receipt(built)
        subprocess.run(
            [str(executable), "--help"],
            cwd=temporary_root,
            env={
                **os.environ,
                "PYTHONPATH": "",
                "PYTHONHOME": "",
            },
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        if output_dir.exists():
            if output_dir.is_symlink() or not output_dir.is_dir():
                raise RuntimeError("desktop backend output must be a real directory")
            shutil.rmtree(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(built), str(output_dir))
        return output_dir / executable.name
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the frozen desktop GUI backend.")
    parser.add_argument(
        "--output-dir",
        default=str(GUI_ROOT / ".desktop-stage" / "backend"),
    )
    parser.add_argument(
        "--work-dir",
        default=str(GUI_ROOT / ".desktop-build" / "pyinstaller"),
    )
    args = parser.parse_args(argv)
    executable = build_backend(Path(args.output_dir), Path(args.work_dir))
    print(json.dumps({"backend": str(executable)}, sort_keys=True))


if __name__ == "__main__":
    main()
