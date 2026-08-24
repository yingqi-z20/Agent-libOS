from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Iterable
import zipfile


ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.5.1"
CHANNEL = "internal-unsigned"
APP_ID = "io.agentlibos.desktop"
MAX_ARCHIVE_FILES = 50_000
MAX_PATH_BYTES = 1_024
FORBIDDEN_PARTS = frozenset(
    {".git", ".venv", "__pycache__", "tests", "test", "fixtures"}
)
FORBIDDEN_SUFFIXES = frozenset(
    {".db", ".env", ".pem", ".pfx", ".p12", ".pyc", ".pyo", ".sqlite"}
)
MACHO_MAGICS = frozenset(
    {
        b"\xca\xfe\xba\xbe",
        b"\xbe\xba\xfe\xca",
        b"\xcf\xfa\xed\xfe",
        b"\xfe\xed\xfa\xcf",
        b"\xce\xfa\xed\xfe",
        b"\xfe\xed\xfa\xce",
    }
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_macho(path: Path) -> bool:
    with path.open("rb") as source:
        return source.read(4) in MACHO_MAGICS


def _release_label(target: str) -> str:
    labels = {
        "darwin-arm64": "macos-arm64",
        "win32-x64": "windows-x64",
        "linux-x64": "linux-x64",
    }
    if target not in labels:
        raise ValueError(f"unsupported desktop target: {target!r}")
    return labels[target]


def _package_names(target: str) -> tuple[str, ...]:
    if target == "darwin-arm64":
        return (
            f"Agent-libOS-{VERSION}-macos-arm64.dmg",
            f"Agent-libOS-{VERSION}-macos-arm64.zip",
        )
    if target == "win32-x64":
        return (
            f"Agent-libOS-Setup-{VERSION}-windows-x64.exe",
            f"Agent-libOS-{VERSION}-windows-x64.zip",
        )
    if target == "linux-x64":
        return (
            f"Agent-libOS-{VERSION}-linux-x64.AppImage",
            f"Agent-libOS-{VERSION}-linux-x64.tar.gz",
        )
    raise ValueError(f"unsupported desktop target: {target!r}")


def _sidecar_names(target: str) -> tuple[str, ...]:
    label = _release_label(target)
    prefix = f"Agent-libOS-{VERSION}-{label}"
    return (
        f"{prefix}-SBOM.cdx.json",
        f"{prefix}-components.json",
        f"{prefix}-THIRD_PARTY_NOTICES.txt",
    )


def _regular_file(path: Path, *, minimum: int = 1, maximum: int | None = None) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"desktop artifact is not a regular file: {path.name}")
    size = path.stat().st_size
    if size < minimum or (maximum is not None and size > maximum):
        raise ValueError(f"desktop artifact size is invalid: {path.name}")


def _detect(root: Path) -> tuple[str, Path, dict[str, Any]]:
    matches = sorted(root.glob(f"Agent-libOS-{VERSION}-*-components.json"))
    if len(matches) != 1:
        raise ValueError("desktop artifact directory must contain one component manifest")
    component_path = matches[0]
    component = _read_json(component_path)
    target = component.get("target")
    if not isinstance(target, str):
        raise ValueError("desktop component manifest target is invalid")
    expected = f"Agent-libOS-{VERSION}-{_release_label(target)}-components.json"
    if component_path.name != expected:
        raise ValueError("desktop component manifest filename does not match its target")
    return target, component_path, component


def _validate_component_manifest(component: dict[str, Any], target: str) -> None:
    if component.get("schema_version") != 1:
        raise ValueError("desktop component manifest schema version is invalid")
    if component.get("distribution_channel") != CHANNEL:
        raise ValueError("desktop distribution must be marked internal-unsigned")
    if component.get("target") != target:
        raise ValueError("desktop component manifest target changed")
    product = component.get("product")
    if product != {"app_id": APP_ID, "name": "Agent libOS", "version": VERSION}:
        raise ValueError("desktop product identity is invalid")
    runtime_components = component.get("components")
    third_party = component.get("third_party_components")
    if not isinstance(runtime_components, dict) or set(runtime_components) != {
        "backend",
        "deno",
        "renderer",
    }:
        raise ValueError("desktop runtime component inventory is incomplete")
    if not isinstance(third_party, list) or not third_party:
        raise ValueError("desktop third-party component inventory is empty")


def _validate_sbom(path: Path, target: str) -> None:
    sbom = _read_json(path)
    if sbom.get("bomFormat") != "CycloneDX" or sbom.get("specVersion") != "1.6":
        raise ValueError("desktop SBOM must be CycloneDX 1.6")
    metadata = sbom.get("metadata")
    application = metadata.get("component") if isinstance(metadata, dict) else None
    if not isinstance(application, dict):
        raise ValueError("desktop SBOM application component is missing")
    if (application.get("name"), application.get("version")) != ("Agent libOS", VERSION):
        raise ValueError("desktop SBOM application identity is invalid")
    properties = application.get("properties")
    observed = {
        item.get("name"): item.get("value")
        for item in properties
        if isinstance(item, dict)
    } if isinstance(properties, list) else {}
    if observed != {
        "agent-libos:distribution-channel": CHANNEL,
        "agent-libos:target": target,
    }:
        raise ValueError("desktop SBOM target/channel properties are invalid")
    components = sbom.get("components")
    if not isinstance(components, list):
        raise ValueError("desktop SBOM components are invalid")
    versions = {
        str(item.get("name")).casefold(): str(item.get("version"))
        for item in components
        if isinstance(item, dict)
    }
    expected = {
        "agent-libos": VERSION,
        "cpython": "3.11.15",
        "deno": "2.9.5",
        "electron": "43.2.0",
        "keyring": "25.7.0",
        "mcp": "2.0.0",
    }
    mismatches = {
        name: (versions.get(name), version)
        for name, version in expected.items()
        if versions.get(name) != version
    }
    if mismatches:
        raise ValueError(f"desktop SBOM exact component versions changed: {mismatches}")


def _validate_checksums(root: Path, required: set[str]) -> None:
    checksum_path = root / "SHA256SUMS"
    _regular_file(checksum_path, maximum=128 * 1024)
    values: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\]+)", line)
        if match is None or match.group(2) in values:
            raise ValueError("SHA256SUMS is not canonical")
        values[match.group(2)] = match.group(1)
    if set(values) != required:
        raise ValueError("SHA256SUMS does not cover the exact desktop release files")
    for name, expected in values.items():
        if _sha256(root / name) != expected:
            raise ValueError(f"desktop artifact checksum mismatch: {name}")


def _safe_archive_path(name: str) -> PurePosixPath:
    if "\\" in name or len(name.encode("utf-8")) > MAX_PATH_BYTES:
        raise ValueError(f"desktop archive contains an unsafe path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"desktop archive contains an unsafe path: {name!r}")
    folded = {part.casefold() for part in path.parts}
    if folded & FORBIDDEN_PARTS:
        raise ValueError(f"desktop archive contains a forbidden path: {name!r}")
    leaf = path.name.casefold()
    normalized = path.as_posix().casefold()
    public_ca_bundle = normalized.endswith(
        "/backend/_internal/certifi/cacert.pem"
    )
    if leaf == ".env" or (
        not public_ca_bundle
        and any(leaf.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES)
    ):
        raise ValueError(f"desktop archive contains mutable/private data: {name!r}")
    if leaf.endswith(".py"):
        if "/backend/_internal/keyring/" not in normalized and not normalized.endswith(
            "/backend/_internal/agent_libos/modules/core.py"
        ):
            raise ValueError(f"desktop archive contains unexpected Python source: {name!r}")
    return path


def _safe_link(path: PurePosixPath, target: str) -> None:
    if not target or "\\" in target:
        raise ValueError(f"desktop archive link target is invalid: {target!r}")
    selected = PurePosixPath(target)
    if selected.is_absolute():
        raise ValueError(f"desktop archive contains an absolute link: {path}")
    stack = list(path.parent.parts)
    for part in selected.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not stack:
                raise ValueError(f"desktop archive link escapes its root: {path}")
            stack.pop()
        else:
            stack.append(part)


def _validate_zip(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if not members or len(members) > MAX_ARCHIVE_FILES:
            raise ValueError(f"desktop ZIP member count is invalid: {path.name}")
        seen: set[PurePosixPath] = set()
        for member in members:
            selected = _safe_archive_path(member.filename.rstrip("/"))
            if selected in seen:
                raise ValueError(f"desktop ZIP contains a duplicate path: {selected}")
            seen.add(selected)
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                _safe_link(selected, archive.read(member).decode("utf-8"))


def _validate_tar(path: Path) -> None:
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
        if not members or len(members) > MAX_ARCHIVE_FILES:
            raise ValueError(f"desktop tar member count is invalid: {path.name}")
        seen: set[PurePosixPath] = set()
        for member in members:
            selected = _safe_archive_path(member.name.rstrip("/"))
            if selected in seen:
                raise ValueError(f"desktop tar contains a duplicate path: {selected}")
            seen.add(selected)
            if member.ischr() or member.isblk() or member.isfifo():
                raise ValueError(f"desktop tar contains a special file: {selected}")
            if member.issym() or member.islnk():
                _safe_link(selected, member.linkname)


def _unpacked_layout(root: Path, target: str) -> tuple[Path, Path, Path, Path]:
    if target == "darwin-arm64":
        app = root / "mac-arm64" / "Agent libOS.app"
        resources = app / "Contents" / "Resources"
        executable = app / "Contents" / "MacOS" / "Agent libOS"
        backend = resources / "backend" / "agent-libos-gui-server"
    elif target == "win32-x64":
        app = root / "win-unpacked"
        resources = app / "resources"
        executable = app / "Agent libOS.exe"
        backend = resources / "backend" / "agent-libos-gui-server.exe"
    elif target == "linux-x64":
        app = root / "linux-unpacked"
        resources = app / "resources"
        executable = app / "agent-libos"
        backend = resources / "backend" / "agent-libos-gui-server"
    else:  # pragma: no cover - target checked before this point
        raise ValueError(target)
    deno = resources / "bin" / ("deno.exe" if target == "win32-x64" else "deno")
    return app, resources, executable, backend if backend else deno


def _scan_resource_tree(resources: Path) -> None:
    if not resources.is_dir() or resources.is_symlink():
        raise ValueError("unpacked Electron resources directory is unavailable")
    required = {
        "app.asar",
        "backend",
        "bin",
        "legal",
        "metadata",
        "renderer",
    }
    if not required.issubset({path.name for path in resources.iterdir()}):
        raise ValueError("unpacked Electron resources are incomplete")
    for selected in resources.rglob("*"):
        relative = selected.relative_to(resources)
        folded = {part.casefold() for part in relative.parts}
        if folded & FORBIDDEN_PARTS:
            raise ValueError(f"unpacked desktop resources contain {relative}")
        if selected.is_file():
            leaf = selected.name.casefold()
            normalized = relative.as_posix().casefold()
            public_ca_bundle = normalized == "backend/_internal/certifi/cacert.pem"
            if leaf == ".env" or (
                not public_ca_bundle
                and any(leaf.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES)
            ):
                raise ValueError(f"unpacked desktop resources contain mutable/private data: {relative}")
            if leaf.endswith(".py"):
                if not normalized.startswith("backend/_internal/keyring/") and normalized != (
                    "backend/_internal/agent_libos/modules/core.py"
                ):
                    raise ValueError(f"unpacked desktop resources contain unexpected Python source: {relative}")


def _contains_bytes(path: Path, needles: Iterable[bytes]) -> bytes | None:
    selected = tuple(needle for needle in needles if needle)
    if not selected:
        return None
    overlap = max(len(needle) for needle in selected) - 1
    prior = b""
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value = prior + block
            for needle in selected:
                if needle in value:
                    return needle
            prior = value[-overlap:] if overlap > 0 else b""
    return None


def _scan_checkout_references(resources: Path) -> None:
    needles = (
        str(ROOT).encode(),
        str(ROOT).replace("/", "\\").encode(),
        b"/home/runner/work/Agent-libOS",
        b"\\a\\Agent-libOS",
        b"/.venv/",
        b"\\.venv\\",
    )
    for selected in resources.rglob("*"):
        if not selected.is_file() or selected.is_symlink():
            continue
        found = _contains_bytes(selected, needles)
        if found is not None:
            raise ValueError(
                f"desktop resource contains a checkout/development path in {selected.relative_to(resources)}"
            )


def _validate_internal_metadata(
    resources: Path,
    component: dict[str, Any],
    target: str,
) -> None:
    internal_component = _read_json(resources / "metadata" / "desktop-component-manifest.json")
    if internal_component != component:
        raise ValueError("external component inventory differs from the packaged inventory")
    runtime = _read_json(resources / "metadata" / "runtime-manifest.json")
    if runtime.get("distribution_channel") != CHANNEL:
        raise ValueError("packaged runtime channel is invalid")
    if runtime.get("product") != {"app_id": APP_ID, "name": "Agent libOS", "version": VERSION}:
        raise ValueError("packaged runtime identity is invalid")
    toolchain = runtime.get("toolchain")
    if toolchain != {
        "deno": "2.9.5",
        "electron": "43.2.0",
        "electron_builder": "26.15.3",
        "keyring": "25.7.0",
        "mcp": "2.0.0",
        "node": "24.15.0",
        "pyinstaller": "6.21.0",
        "python": "3.11.15",
    }:
        raise ValueError("packaged desktop toolchain versions changed")
    paths = component["components"]
    for name in ("backend", "deno", "renderer"):
        item = paths[name]
        if not isinstance(item, dict):
            raise ValueError(f"desktop {name} component metadata is invalid")
        relative = item.get("path")
        expected_hash = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError(f"desktop {name} component metadata is invalid")
        selected = resources / PurePosixPath(relative)
        _regular_file(selected)
        if _sha256(selected) != expected_hash and not (
            target == "darwin-arm64" and _is_macho(selected)
        ):
            raise ValueError(f"packaged desktop {name} hash changed")
    if target != component["target"]:
        raise ValueError("packaged desktop target changed")


def _validate_macos(app: Path) -> None:
    subprocess.run(
        ["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    plist = app / "Contents" / "Info.plist"
    identifier = subprocess.run(
        ["/usr/libexec/PlistBuddy", "-c", "Print :CFBundleIdentifier", str(plist)],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    version = subprocess.run(
        ["/usr/libexec/PlistBuddy", "-c", "Print :CFBundleShortVersionString", str(plist)],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    if (identifier, version) != (APP_ID, VERSION):
        raise ValueError("macOS bundle identity/version is invalid")
    for selected in app.rglob("*"):
        if not selected.is_file() or selected.is_symlink():
            continue
        with selected.open("rb") as source:
            if source.read(4) not in MACHO_MAGICS:
                continue
        subprocess.run(
            ["codesign", "--verify", "--strict", "--verbose=2", str(selected)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        linked = subprocess.run(
            # ``-m`` disables archive(member) parsing; Electron helper names
            # legitimately contain parentheses (for example ``(GPU)``).
            ["otool", "-m", "-L", str(selected)],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.splitlines()[1:]
        for line in linked:
            dependency = line.strip().split(" ", 1)[0]
            if not dependency.startswith(
                ("@", "./", "/System/Library/", "/usr/lib/")
            ):
                raise ValueError(f"Mach-O links outside the OS/bundle boundary: {dependency}")


def _validate_linux(app: Path) -> None:
    for selected in app.rglob("*"):
        if not selected.is_file() or selected.is_symlink():
            continue
        try:
            with selected.open("rb") as source:
                magic = source.read(4)
        except OSError:
            continue
        if magic != b"\x7fELF":
            continue
        result = subprocess.run(
            ["ldd", str(selected)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if "not found" in result.stdout:
            raise ValueError(f"ELF dependency is unavailable for {selected.name}")
        for match in re.findall(r"=>\s+(/\S+)", result.stdout):
            if not match.startswith(("/lib/", "/lib64/", "/usr/lib/")):
                raise ValueError(f"ELF links outside the OS boundary: {match}")


def validate(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("desktop artifact path must be a real directory")
    unpublished = sorted(
        path.name
        for path in root.iterdir()
        if path.name.endswith(".blockmap")
        or path.name in {"builder-debug.yml", "builder-effective-config.yaml"}
        or path.name.startswith(".icon-")
    )
    if unpublished:
        raise ValueError(
            f"desktop output contains unpublished update/debug metadata: {unpublished}"
        )
    target, component_path, component = _detect(root)
    _validate_component_manifest(component, target)
    package_names = _package_names(target)
    sidecar_names = _sidecar_names(target)
    for name in (*package_names, *sidecar_names):
        _regular_file(root / name, minimum=1, maximum=4 * 1024 * 1024 * 1024)
    required = set((*package_names, *sidecar_names))
    _validate_checksums(root, required)
    _validate_sbom(root / sidecar_names[0], target)
    notice = (root / sidecar_names[2]).read_text(encoding="utf-8")
    folded_notice = notice.casefold()
    for marker in ("Deno 2.9.5", "Electron 43.2.0", "keyring 25.7.0", "mcp 2.0.0"):
        if marker.casefold() not in folded_notice:
            raise ValueError(f"third-party notice is missing {marker}")
    portable = root / package_names[1]
    if portable.suffix == ".zip":
        _validate_zip(portable)
    else:
        _validate_tar(portable)
    app, resources, executable, _backend = _unpacked_layout(root, target)
    _regular_file(executable)
    _scan_resource_tree(resources)
    _scan_checkout_references(resources)
    _validate_internal_metadata(resources, component, target)
    if target == "darwin-arm64":
        _validate_macos(app)
    elif target == "linux-x64":
        _validate_linux(app)
    return {
        "channel": CHANNEL,
        "files": sorted(required),
        "target": target,
        "version": VERSION,
        "component_manifest": component_path.name,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate one native Agent libOS internal desktop artifact set."
    )
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        result = validate(args.artifact_dir)
    except (OSError, ValueError, subprocess.SubprocessError, tarfile.TarError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
