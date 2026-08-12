from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE = ROOT / "gui" / ".desktop-stage"
DEFAULT_OUTPUT = ROOT / "desktop-dist"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path.name} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_directory(path: Path, *, label: str) -> Path:
    selected = path.expanduser().resolve()
    if selected in {Path(selected.anchor), ROOT, ROOT / "gui"}:
        raise RuntimeError(f"refusing broad {label} directory")
    if selected.exists() and (selected.is_symlink() or not selected.is_dir()):
        raise RuntimeError(f"{label} must be a real directory")
    return selected


def _release_label(target: str) -> str:
    labels = {
        "darwin-arm64": "macos-arm64",
        "win32-x64": "windows-x64",
        "linux-x64": "linux-x64",
    }
    try:
        return labels[target]
    except KeyError as exc:
        raise RuntimeError(f"unsupported desktop target {target!r}") from exc


def _package_names(version: str, target: str) -> tuple[str, ...]:
    if target == "darwin-arm64":
        return (
            f"Agent-libOS-{version}-macos-arm64.dmg",
            f"Agent-libOS-{version}-macos-arm64.zip",
        )
    if target == "win32-x64":
        return (
            f"Agent-libOS-Setup-{version}-windows-x64.exe",
            f"Agent-libOS-{version}-windows-x64.zip",
        )
    if target == "linux-x64":
        return (
            f"Agent-libOS-{version}-linux-x64.AppImage",
            f"Agent-libOS-{version}-linux-x64.tar.gz",
        )
    raise RuntimeError(f"unsupported desktop target {target!r}")


def _atomic_copy(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"desktop release input is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", dir=destination.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        shutil.copyfile(source, temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _remove_unpublished_builder_metadata(output_dir: Path) -> None:
    """Remove update/debug intermediates that are not desktop release inputs."""

    for selected in output_dir.glob("*.blockmap"):
        if selected.is_file() and not selected.is_symlink():
            selected.unlink()
    for name in ("builder-debug.yml", "builder-effective-config.yaml"):
        selected = output_dir / name
        if selected.is_file() and not selected.is_symlink():
            selected.unlink()
    for selected in output_dir.glob(".icon-*"):
        if selected.is_dir() and not selected.is_symlink():
            shutil.rmtree(selected)


def finalize(output_dir: Path, stage_root: Path) -> list[Path]:
    output_dir = _safe_directory(output_dir, label="desktop output")
    stage_root = _safe_directory(stage_root, label="desktop stage")
    _remove_unpublished_builder_metadata(output_dir)
    metadata_root = stage_root / "metadata"
    component_manifest = _read_json(metadata_root / "desktop-component-manifest.json")
    target = component_manifest.get("target")
    product = component_manifest.get("product")
    if not isinstance(target, str) or not isinstance(product, dict):
        raise RuntimeError("desktop component manifest is malformed")
    version = product.get("version")
    if not isinstance(version, str) or not version:
        raise RuntimeError("desktop component manifest version is invalid")
    packages = [output_dir / name for name in _package_names(version, target)]
    for package in packages:
        if not package.is_file() or package.is_symlink() or package.stat().st_size == 0:
            raise RuntimeError(f"desktop package is missing: {package.name}")

    label = _release_label(target)
    external = {
        output_dir / f"Agent-libOS-{version}-{label}-SBOM.cdx.json": (
            metadata_root / "agent-libos-desktop.cdx.json"
        ),
        output_dir / f"Agent-libOS-{version}-{label}-components.json": (
            metadata_root / "desktop-component-manifest.json"
        ),
        output_dir / f"Agent-libOS-{version}-{label}-THIRD_PARTY_NOTICES.txt": (
            stage_root / "legal" / "THIRD_PARTY_NOTICES.txt"
        ),
    }
    for destination, source in external.items():
        _atomic_copy(source, destination)

    checksum_targets = sorted([*packages, *external], key=lambda path: path.name)
    checksum_text = "".join(
        f"{_sha256(path)}  {path.name}\n" for path in checksum_targets
    )
    checksum_path = output_dir / "SHA256SUMS"
    temporary = output_dir / ".SHA256SUMS.tmp"
    temporary.write_text(checksum_text, encoding="ascii")
    temporary.replace(checksum_path)
    return [*checksum_targets, checksum_path]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Publish desktop SBOM, component, notice, and checksum sidecars."
    )
    parser.add_argument("output_dir", nargs="?", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--stage-root", default=str(DEFAULT_STAGE))
    args = parser.parse_args(argv)
    files = finalize(Path(args.output_dir), Path(args.stage_root))
    print(json.dumps({"artifacts": [str(path) for path in files]}, sort_keys=True))


if __name__ == "__main__":
    main()
