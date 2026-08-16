from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

try:
    import check_desktop_artifacts as checker
    from smoke_desktop_bundle import _electron_smoke
except ModuleNotFoundError:  # imported as scripts.verify_desktop_installers
    from scripts import check_desktop_artifacts as checker
    from scripts.smoke_desktop_bundle import _electron_smoke


VERSION = "1.5.1"


def _single(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise RuntimeError(f"expected one {label}, found {len(paths)}")
    return paths[0]


def _macos(root: Path, temporary: Path, *, window: bool) -> dict[str, object]:
    image = root / f"Agent-libOS-{VERSION}-macos-arm64.dmg"
    mount = temporary / "mount"
    mount.mkdir()
    subprocess.run(
        ["hdiutil", "attach", "-nobrowse", "-readonly", "-mountpoint", str(mount), str(image)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    copied = temporary / "installed" / "Agent libOS.app"
    try:
        source = _single(list(mount.glob("*.app")), "application in DMG")
        copied.parent.mkdir()
        shutil.copytree(source, copied, symlinks=True)
    finally:
        subprocess.run(
            ["hdiutil", "detach", str(mount)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    executable = copied / "Contents" / "MacOS" / "Agent libOS"
    backend = copied / "Contents" / "Resources" / "backend" / "agent-libos-gui-server"
    installed = _electron_smoke(
        executable,
        backend,
        temporary / "mac-smoke",
        window=window,
    )

    archive = root / f"Agent-libOS-{VERSION}-macos-arm64.zip"
    checker._validate_zip(archive)
    portable_root = temporary / "portable"
    portable_root.mkdir()
    # ditto preserves executable bits, framework symlinks, and macOS bundle
    # metadata while expanding an Electron Builder ZIP.
    subprocess.run(
        ["ditto", "-x", "-k", str(archive), str(portable_root)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    portable_app = _single(list(portable_root.glob("*.app")), "application in ZIP")
    portable_executable = portable_app / "Contents" / "MacOS" / "Agent libOS"
    portable_backend = (
        portable_app
        / "Contents"
        / "Resources"
        / "backend"
        / "agent-libos-gui-server"
    )
    portable = _electron_smoke(
        portable_executable,
        portable_backend,
        temporary / "portable-smoke",
        window=window,
    )
    return {
        "dmg": "mounted-copied-launched",
        "installed": installed,
        "zip": portable,
    }


def _windows(root: Path, temporary: Path, *, window: bool) -> dict[str, object]:
    installer = root / f"Agent-libOS-Setup-{VERSION}-windows-x64.exe"
    install_root = temporary / "installed"
    subprocess.run(
        [str(installer), "/S", f"/D={install_root}"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    installed_exe = _single(
        [path for path in install_root.rglob("Agent libOS.exe") if path.is_file()],
        "installed Electron executable",
    )
    installed_backend = installed_exe.parent / "resources" / "backend" / "agent-libos-gui-server.exe"
    installed = _electron_smoke(
        installed_exe,
        installed_backend,
        temporary / "installed-smoke",
        window=window,
    )
    uninstaller = _single(
        [path for path in install_root.rglob("*.exe") if path.name.casefold().startswith("uninstall")],
        "NSIS uninstaller",
    )
    subprocess.run(
        [str(uninstaller), "/S"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    if installed_exe.exists():
        raise RuntimeError("NSIS uninstall retained the installed application")

    archive = root / f"Agent-libOS-{VERSION}-windows-x64.zip"
    checker._validate_zip(archive)
    portable_root = temporary / "portable"
    portable_root.mkdir()
    with zipfile.ZipFile(archive) as package:
        package.extractall(portable_root)
    portable_exe = _single(list(portable_root.rglob("Agent libOS.exe")), "portable Electron executable")
    portable_backend = portable_exe.parent / "resources" / "backend" / "agent-libos-gui-server.exe"
    portable = _electron_smoke(
        portable_exe,
        portable_backend,
        temporary / "portable-smoke",
        window=window,
    )
    return {"nsis": installed, "uninstalled": True, "zip": portable}


def _linux(root: Path, temporary: Path, *, window: bool) -> dict[str, object]:
    image = root / f"Agent-libOS-{VERSION}-linux-x64.AppImage"
    image.chmod(image.stat().st_mode | 0o111)
    appimage = _electron_smoke(
        image,
        None,
        temporary / "appimage-smoke",
        window=window,
        extra_env={"APPIMAGE_EXTRACT_AND_RUN": "1"},
    )

    archive = root / f"Agent-libOS-{VERSION}-linux-x64.tar.gz"
    checker._validate_tar(archive)
    portable_root = temporary / "portable"
    portable_root.mkdir()
    with tarfile.open(archive, mode="r:gz") as package:
        package.extractall(portable_root, filter="data")
    executables = [
        path
        for path in portable_root.rglob("agent-libos")
        if path.is_file() and path.parent.name != "backend"
    ]
    portable_exe = _single(executables, "portable Linux Electron executable")
    portable_backend = portable_exe.parent / "resources" / "backend" / "agent-libos-gui-server"
    portable = _electron_smoke(
        portable_exe,
        portable_backend,
        temporary / "portable-smoke",
        window=window,
    )
    return {"appimage": appimage, "tar_gz": portable}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Exercise native desktop installers and portable archives.")
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args(argv)
    root = args.artifact_dir.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        parser.error("artifact_dir must be a real directory")
    with tempfile.TemporaryDirectory(prefix="agent-libos-desktop-installer-") as selected:
        temporary = Path(selected)
        if sys.platform == "darwin":
            result = _macos(root, temporary, window=not args.no_window)
            target = "darwin-arm64"
        elif sys.platform == "win32":
            result = _windows(root, temporary, window=not args.no_window)
            target = "win32-x64"
        elif sys.platform.startswith("linux"):
            result = _linux(root, temporary, window=not args.no_window)
            target = "linux-x64"
        else:
            parser.error(f"unsupported installer platform: {sys.platform}")
    print(json.dumps({"installers": result, "target": target}, sort_keys=True))


if __name__ == "__main__":
    main()
