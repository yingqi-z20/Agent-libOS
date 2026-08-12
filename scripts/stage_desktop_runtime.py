from __future__ import annotations

import argparse
from email.message import Message
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any
import urllib.parse
import urllib.request
import zipfile

from packaging.requirements import Requirement

try:
    from build_desktop_backend import build_backend
except ModuleNotFoundError:  # imported as scripts.stage_desktop_runtime in tests
    from scripts.build_desktop_backend import build_backend


ROOT = Path(__file__).resolve().parents[1]
GUI_ROOT = ROOT / "gui"
MANIFEST_PATH = ROOT / "desktop" / "runtime-manifest.json"
MAX_DOWNLOAD_BYTES = 256 * 1024 * 1024
ALLOWED_DOWNLOAD_HOSTS = frozenset(
    {"github.com", "release-assets.githubusercontent.com", "raw.githubusercontent.com"}
)


def _read_manifest() -> dict[str, Any]:
    value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("desktop runtime manifest must be an object")
    return value


def _target_key() -> str:
    machine = platform.machine().lower()
    if sys.platform == "darwin" and machine in {"arm64", "aarch64"}:
        return "darwin-arm64"
    if sys.platform == "win32" and machine in {"amd64", "x86_64"}:
        return "win32-x64"
    if sys.platform.startswith("linux") and machine in {"amd64", "x86_64"}:
        return "linux-x64"
    raise RuntimeError(f"unsupported desktop target: {sys.platform}-{machine}")


def _safe_stage_path(value: str | Path) -> Path:
    selected = Path(value).expanduser().resolve()
    if selected in {Path(selected.anchor), ROOT, GUI_ROOT}:
        raise RuntimeError("refusing broad desktop stage path")
    return selected


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _download(url: str, expected_sha256: str, *, max_bytes: int = MAX_DOWNLOAD_BYTES) -> bytes:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_DOWNLOAD_HOSTS:
        raise RuntimeError("desktop component URL is not an approved HTTPS origin")
    request = urllib.request.Request(url, headers={"User-Agent": "agent-libos-desktop-builder/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        final = urllib.parse.urlsplit(response.geturl())
        if final.scheme != "https" or final.hostname not in ALLOWED_DOWNLOAD_HOSTS:
            raise RuntimeError("desktop component redirected outside approved HTTPS origins")
        raw_length = response.headers.get("Content-Length")
        if raw_length is not None and int(raw_length) > max_bytes:
            raise RuntimeError("desktop component exceeds the download limit")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = response.read(min(1024 * 1024, max_bytes + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise RuntimeError("desktop component exceeds the download limit")
            chunks.append(chunk)
    value = b"".join(chunks)
    if _sha256(value) != expected_sha256:
        raise RuntimeError("desktop component SHA-256 mismatch")
    return value


def _stage_deno(stage_root: Path, manifest: dict[str, Any], target: str) -> Path:
    deno = manifest["deno"]
    selected = deno["targets"][target]
    asset = selected["asset"]
    url = f"https://github.com/denoland/deno/releases/download/v{deno['version']}/{asset}"
    archive = _download(url, selected["sha256"])
    with zipfile.ZipFile(__import__("io").BytesIO(archive)) as package:
        members = package.infolist()
        expected_name = "deno.exe" if target.startswith("win32-") else "deno"
        if len(members) != 1 or members[0].filename != expected_name:
            raise RuntimeError("Deno archive does not contain exactly the expected executable")
        member = members[0]
        if member.is_dir() or member.file_size <= 0 or member.file_size > MAX_DOWNLOAD_BYTES:
            raise RuntimeError("Deno executable entry is invalid")
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise RuntimeError("Deno archive must not contain a symbolic link")
        executable_bytes = package.read(member)
    bin_root = stage_root / "bin"
    bin_root.mkdir(parents=True, exist_ok=True)
    executable = bin_root / expected_name
    executable.write_bytes(executable_bytes)
    if os.name != "nt":
        executable.chmod(0o755)
    version = subprocess.run(
        [str(executable), "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    ).stdout.splitlines()[0]
    if version != f"deno {deno['version']} (stable, release, {asset.removeprefix('deno-').removesuffix('.zip')})":
        raise RuntimeError(f"unexpected bundled Deno version line: {version}")
    return executable


def _normal_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _runtime_python_distributions() -> list[importlib.metadata.Distribution]:
    pending = ["agent-libos", "keyring", "mcp"]
    selected: dict[str, importlib.metadata.Distribution] = {}
    while pending:
        name = _normal_name(pending.pop())
        if name in selected:
            continue
        distribution = importlib.metadata.distribution(name)
        selected[name] = distribution
        for raw in distribution.requires or ():
            requirement = Requirement(raw)
            if requirement.marker is not None and not requirement.marker.evaluate():
                continue
            pending.append(requirement.name)
    return [selected[name] for name in sorted(selected)]


def _license_expression(metadata: Message) -> str:
    expression = metadata.get("License-Expression") or metadata.get("License")
    if expression and expression.strip() and expression.strip().upper() != "UNKNOWN":
        return expression.strip()
    classifiers = [
        item.removeprefix("License :: ").strip()
        for item in metadata.get_all("Classifier", [])
        if item.startswith("License :: ")
    ]
    return "; ".join(classifiers) or "NOASSERTION"


def _python_components(legal_root: Path) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    license_root = legal_root / "python"
    for distribution in _runtime_python_distributions():
        metadata = distribution.metadata
        name = metadata["Name"] or "unknown"
        version = distribution.version
        license_expression = _license_expression(metadata)
        components.append(
            {
                "type": "library",
                "name": name,
                "version": version,
                "purl": f"pkg:pypi/{_normal_name(name)}@{version}",
                "licenses": [{"expression": license_expression}],
            }
        )
        copied = 0
        for candidate in distribution.files or ():
            filename = Path(str(candidate)).name
            if not re.match(r"(?i)^(license|copying|notice)(\..*)?$", filename):
                continue
            source = Path(distribution.locate_file(candidate)).resolve()
            if not source.is_file() or source.stat().st_size > 2 * 1024 * 1024:
                continue
            target = license_root / f"{_normal_name(name)}-{copied}-{filename}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied += 1
    return components


def _node_components() -> list[dict[str, Any]]:
    lock = json.loads((GUI_ROOT / "package-lock.json").read_text(encoding="utf-8"))
    packages = lock["packages"]
    root = packages[""]
    pending = sorted({*root.get("dependencies", {}), "electron"})
    selected: set[str] = set()
    components: list[dict[str, Any]] = []
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        package = packages.get(f"node_modules/{name}")
        if not isinstance(package, dict):
            raise RuntimeError(f"desktop npm dependency is missing from lockfile: {name}")
        selected.add(name)
        version = package.get("version")
        license_name = package.get("license", "NOASSERTION")
        if not isinstance(version, str) or not version:
            raise RuntimeError(f"desktop npm dependency has no version: {name}")
        components.append(
            {
                "type": "library",
                "name": name,
                "version": version,
                "purl": f"pkg:npm/{urllib.parse.quote(name, safe='@/')}@{version}",
                "licenses": [{"expression": str(license_name)}],
            }
        )
        pending.extend(package.get("dependencies", {}).keys())
        pending.extend(package.get("optionalDependencies", {}).keys())
    return sorted(components, key=lambda item: (item["name"], item["version"]))


def _stage_legal(stage_root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    legal_root = stage_root / "legal"
    legal_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "LICENSE", legal_root / "Agent-libOS-LICENSE.txt")
    deno_license = manifest["deno"]["license"]
    (legal_root / "Deno-LICENSE.txt").write_bytes(
        _download(deno_license["url"], deno_license["sha256"], max_bytes=2 * 1024 * 1024)
    )
    components = _python_components(legal_root) + _node_components()
    components.append(
        {
            "type": "platform",
            "name": "CPython",
            "version": manifest["toolchain"]["python"],
            "purl": f"pkg:generic/cpython@{manifest['toolchain']['python']}",
            "licenses": [{"expression": "Python-2.0"}],
        }
    )
    components.append(
        {
            "type": "application",
            "name": "Deno",
            "version": manifest["deno"]["version"],
            "purl": f"pkg:generic/deno@{manifest['deno']['version']}",
            "licenses": [{"expression": "MIT"}],
        }
    )
    components = sorted(components, key=lambda item: (item["name"].casefold(), item["version"]))
    notices = [
        "Agent libOS desktop internal distribution",
        "",
        "Bundled third-party components:",
    ]
    for component in components:
        expression = component["licenses"][0]["expression"]
        notices.append(f"- {component['name']} {component['version']}: {expression}")
    notices.extend(
        [
            "",
            "This inventory is generated from the frozen Python environment and npm lockfile.",
            "License texts shipped by Python distributions are under legal/python/.",
            "Deno and Agent libOS license texts are included beside this notice.",
        ]
    )
    (legal_root / "THIRD_PARTY_NOTICES.txt").write_text("\n".join(notices) + "\n", encoding="utf-8")
    return components


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stage_metadata(
    stage_root: Path,
    manifest: dict[str, Any],
    target: str,
    backend: Path,
    deno: Path,
    components: list[dict[str, Any]],
) -> None:
    metadata_root = stage_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MANIFEST_PATH, metadata_root / "runtime-manifest.json")
    product = manifest["product"]
    component_manifest = {
        "schema_version": 1,
        "distribution_channel": manifest["distribution_channel"],
        "target": target,
        "product": product,
        "third_party_components": components,
        "components": {
            "backend": {
                "path": f"backend/{backend.name}",
                "sha256": _file_sha256(backend),
            },
            "deno": {
                "path": f"bin/{deno.name}",
                "version": manifest["deno"]["version"],
                "sha256": _file_sha256(deno),
            },
            "renderer": {
                "path": "renderer/index.html",
                "sha256": _file_sha256(stage_root / "renderer" / "index.html"),
            },
        },
    }
    (metadata_root / "desktop-component-manifest.json").write_text(
        json.dumps(component_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": "urn:uuid:00000000-0000-4000-8000-000000000150",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": product["name"],
                "version": product["version"],
                "properties": [
                    {"name": "agent-libos:distribution-channel", "value": manifest["distribution_channel"]},
                    {"name": "agent-libos:target", "value": target},
                ],
            }
        },
        "components": components,
    }
    (metadata_root / "agent-libos-desktop.cdx.json").write_text(
        json.dumps(sbom, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def stage(stage_root: Path, work_root: Path) -> Path:
    manifest = _read_manifest()
    target = _target_key()
    stage_root = _safe_stage_path(stage_root)
    work_root = _safe_stage_path(work_root)
    if stage_root.exists():
        if stage_root.is_symlink() or not stage_root.is_dir():
            raise RuntimeError("desktop stage must be a real directory")
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True, mode=0o700)
    renderer = GUI_ROOT / "dist"
    if not (renderer / "index.html").is_file():
        raise RuntimeError("GUI production renderer has not been built")
    shutil.copytree(renderer, stage_root / "renderer")
    backend = build_backend(stage_root / "backend", work_root / "pyinstaller")
    deno = _stage_deno(stage_root, manifest, target)
    components = _stage_legal(stage_root, manifest)
    _stage_metadata(stage_root, manifest, target, backend, deno, components)
    return stage_root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stage the self-contained desktop Runtime.")
    parser.add_argument("--stage-root", default=str(GUI_ROOT / ".desktop-stage"))
    parser.add_argument("--work-root", default=str(GUI_ROOT / ".desktop-build"))
    args = parser.parse_args(argv)
    selected = stage(Path(args.stage_root), Path(args.work_root))
    print(json.dumps({"stage": str(selected), "target": _target_key()}, sort_keys=True))


if __name__ == "__main__":
    main()
