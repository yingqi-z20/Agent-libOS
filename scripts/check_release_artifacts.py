from __future__ import annotations

import argparse
import ast
from email import policy
from email.parser import BytesParser
import json
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PROJECT_NAME = "agent-libos"
ARCHIVE_NAME = "agent_libos"
BUILTIN_SKILL_IDS = (
    "agent-libos-skill-navigation",
    "agent-libos-authority-basics",
    "agent-libos-capability-delegation",
    "agent-libos-human-collaboration",
    "agent-libos-runtime-session",
    "agent-libos-workspace-navigation",
    "agent-libos-workspace-editing",
    "agent-libos-command-execution",
    "agent-libos-test-log-analysis",
    "agent-libos-tool-protocol-diagnostics",
    "agent-libos-object-memory",
    "agent-libos-object-file-transfer",
    "agent-libos-object-tasks",
    "agent-libos-child-processes",
    "agent-libos-checkpoints",
    "agent-libos-agent-images",
    "agent-libos-jit-tool-authoring",
    "agent-libos-jsonrpc",
    "agent-libos-mcp",
    "agent-libos-git-inspection",
    "agent-libos-git-change-recording",
    "agent-libos-git-branches-worktrees",
    "agent-libos-git-integration-recovery",
    "agent-libos-git-patch-objects",
    "agent-libos-git-remotes",
    "agent-libos-git-pull-requests",
)
BUILTIN_SKILL_ARCHIVE_PATHS = frozenset(
    f"agent_libos/skills/builtin/{skill_id}/SKILL.md"
    for skill_id in BUILTIN_SKILL_IDS
)
_BUILTIN_SKILL_MAX_FILE_BYTES = 24 * 1_024
_BUILTIN_SKILL_MAX_INSTRUCTION_BYTES = 16 * 1_024
_BUILTIN_SKILL_FRONTMATTER_FIELDS = frozenset(
    {"name", "description", "allowed-tools"}
)
_BUILTIN_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]*$")
WHEEL_REQUIRED_FILES = frozenset(
    {
        "agent_libos/__init__.py",
        "agent_libos/__main__.py",
        "agent_libos/api/cli.py",
        "agent_libos/api/gui/server.py",
    }
) | BUILTIN_SKILL_ARCHIVE_PATHS
SDIST_REQUIRED_FILES = frozenset(
    {
        "LICENSE",
        "PKG-INFO",
        "README.md",
        "pyproject.toml",
        "agent_libos/__init__.py",
        "config.yaml",
        "modules/pty/module.yaml",
        "modules/pty/pty_module.py",
        "skills/swe-agent/SKILL.md",
        "images/mini-swe-agent/IMAGE.yaml",
        "docs/release_status.md",
        "tests/invariants.yaml",
        "scripts/check_release_artifacts.py",
    }
) | BUILTIN_SKILL_ARCHIVE_PATHS
SDIST_FORBIDDEN_PARTS = frozenset(
    {
        ".benchmark_runs",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "dist-electron",
        "node_modules",
    }
)
SDIST_FORBIDDEN_SUFFIXES = frozenset({".db", ".pyc", ".pyo", ".sqlite"})
ALLOWED_SECRET_FIXTURES = frozenset(
    {
        "benchmarks/runtime_safety/fixtures/basic_repo/.env",
        "benchmarks/runtime_safety/fixtures/basic_repo/config/private.key",
        "benchmarks/runtime_safety/fixtures/basic_repo/secrets/token.txt",
    }
)


def _python_package_version(root: Path) -> str:
    tree = ast.parse((root / "agent_libos" / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
    raise ValueError("agent_libos.__version__ must be a string literal")


def release_versions(root: Path = ROOT) -> dict[str, str]:
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    uv_lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    project_lock = next(
        (
            package
            for package in uv_lock.get("package", [])
            if package.get("name") == PROJECT_NAME and package.get("source", {}).get("editable") == "."
        ),
        None,
    )
    if project_lock is None:
        raise ValueError(f"uv.lock does not contain the editable {PROJECT_NAME} package")
    versions = {
        "pyproject.toml": str(pyproject["project"]["version"]),
        "agent_libos/__init__.py": _python_package_version(root),
        "uv.lock": str(project_lock["version"]),
    }
    gui_package_path = root / "gui" / "package.json"
    gui_lock_path = root / "gui" / "package-lock.json"
    if gui_package_path.exists() != gui_lock_path.exists():
        raise ValueError("GUI package metadata must include both package.json and package-lock.json")
    if gui_package_path.exists():
        gui_package = json.loads(gui_package_path.read_text(encoding="utf-8"))
        gui_lock = json.loads(gui_lock_path.read_text(encoding="utf-8"))
        gui_lock_root = gui_lock.get("packages", {}).get("", {})
        versions.update(
            {
                "gui/package.json": str(gui_package["version"]),
                "gui/package-lock.json": str(gui_lock["version"]),
                "gui/package-lock.json packages root": str(gui_lock_root["version"]),
            }
        )
    return versions


def validate_version_alignment(root: Path = ROOT) -> str:
    versions = release_versions(root)
    selected = versions["pyproject.toml"]
    mismatches = {source: version for source, version in versions.items() if version != selected}
    if mismatches:
        details = ", ".join(f"{source}={version}" for source, version in mismatches.items())
        raise ValueError(f"release version identifiers do not match {selected}: {details}")
    return selected


def _safe_archive_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"archive contains unsafe path: {name}")
    return path


def _single_artifact(artifact_dir: Path, pattern: str, kind: str) -> Path:
    matches = sorted(artifact_dir.glob(pattern))
    if len(matches) != 1:
        rendered = ", ".join(path.name for path in matches) or "none"
        raise ValueError(f"expected exactly one {kind} matching {pattern}, found: {rendered}")
    return matches[0]


def _decode_builtin_frontmatter_scalar(raw: str, *, field: str) -> str:
    value = raw.strip()
    if not value:
        raise ValueError(f"built-in Skill {field} must be non-empty")
    if value.startswith('"'):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid quoted built-in Skill {field}") from exc
        if not isinstance(decoded, str) or not decoded.strip():
            raise ValueError(f"built-in Skill {field} must be a non-empty string")
        return decoded.strip()
    if value.startswith("'"):
        if len(value) < 2 or not value.endswith("'"):
            raise ValueError(f"invalid quoted built-in Skill {field}")
        decoded = value[1:-1].replace("''", "'")
        if not decoded.strip():
            raise ValueError(f"built-in Skill {field} must be a non-empty string")
        return decoded.strip()
    if re.search(r":(?:\s|$)", value):
        raise ValueError(
            f"plain built-in Skill {field} contains a YAML mapping delimiter"
        )
    return value


def _parse_builtin_skill_archive_entry(raw: bytes, *, expected_id: str) -> tuple[str, ...]:
    """Parse the dependency-free built-in Skill subset used during release validation."""

    if len(raw) > _BUILTIN_SKILL_MAX_FILE_BYTES:
        raise ValueError(f"built-in Skill {expected_id} exceeds archive size limit")
    try:
        text = raw.decode("utf-8").replace("\r\n", "\n")
    except UnicodeDecodeError as exc:
        raise ValueError(f"built-in Skill {expected_id} is not UTF-8") from exc
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"built-in Skill {expected_id} lacks YAML frontmatter")
    end_index = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
        None,
    )
    if end_index is None:
        raise ValueError(f"built-in Skill {expected_id} has unterminated frontmatter")

    fields: dict[str, str] = {}
    for line in lines[1:end_index]:
        if not line.strip() or line[:1].isspace():
            raise ValueError(
                f"built-in Skill {expected_id} frontmatter must use scalar fields"
            )
        key, separator, value = line.partition(":")
        key = key.strip()
        if not separator or key not in _BUILTIN_SKILL_FRONTMATTER_FIELDS:
            raise ValueError(f"built-in Skill {expected_id} has invalid frontmatter")
        if key in fields:
            raise ValueError(f"built-in Skill {expected_id} repeats frontmatter field {key}")
        fields[key] = _decode_builtin_frontmatter_scalar(value, field=key)
    if set(fields) != _BUILTIN_SKILL_FRONTMATTER_FIELDS:
        missing = sorted(_BUILTIN_SKILL_FRONTMATTER_FIELDS - set(fields))
        raise ValueError(f"built-in Skill {expected_id} is missing frontmatter fields: {missing}")
    if fields["name"] != expected_id:
        raise ValueError(
            f"built-in Skill archive directory/name mismatch: {expected_id} != {fields['name']}"
        )

    allowed_tools = tuple(fields["allowed-tools"].split())
    if not 1 <= len(allowed_tools) <= 9:
        raise ValueError(f"built-in Skill {expected_id} has invalid allowed-tools count")
    if len(set(allowed_tools)) != len(allowed_tools) or any(
        not _BUILTIN_TOOL_NAME_PATTERN.fullmatch(tool) for tool in allowed_tools
    ):
        raise ValueError(f"built-in Skill {expected_id} has invalid allowed-tools")
    instructions = "\n".join(lines[end_index + 1 :]).lstrip("\n")
    if not instructions.strip():
        raise ValueError(f"built-in Skill {expected_id} has no instructions")
    if len(instructions.encode("utf-8")) > _BUILTIN_SKILL_MAX_INSTRUCTION_BYTES:
        raise ValueError(
            f"built-in Skill {expected_id} instructions exceed archive size limit"
        )
    missing_guidance = [
        tool for tool in allowed_tools if f"`{tool}`" not in instructions
    ]
    if missing_guidance:
        raise ValueError(
            f"built-in Skill {expected_id} does not guide tools: {missing_guidance}"
        )
    return allowed_tools


def _validate_builtin_skill_archive_payloads(payloads: dict[str, bytes]) -> None:
    expected_paths = BUILTIN_SKILL_ARCHIVE_PATHS
    actual_paths = frozenset(payloads)
    if actual_paths != expected_paths:
        raise ValueError(
            "archive built-in Skill catalog mismatch: "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"unexpected={sorted(actual_paths - expected_paths)}"
        )
    owner_by_tool: dict[str, str] = {}
    for skill_id in BUILTIN_SKILL_IDS:
        path = f"agent_libos/skills/builtin/{skill_id}/SKILL.md"
        for tool in _parse_builtin_skill_archive_entry(
            payloads[path],
            expected_id=skill_id,
        ):
            previous = owner_by_tool.get(tool)
            if previous is not None:
                raise ValueError(
                    f"archive built-in tool {tool} is owned by both {previous} and {skill_id}"
                )
            owner_by_tool[tool] = skill_id
    if len(owner_by_tool) != 99:
        raise ValueError(
            f"archive built-in Skill catalog must own exactly 99 tools, found {len(owner_by_tool)}"
        )


def _validate_wheel(wheel_path: Path, version: str) -> None:
    dist_info = f"{ARCHIVE_NAME}-{version}.dist-info"
    with zipfile.ZipFile(wheel_path) as archive:
        for item in archive.infolist():
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError(f"wheel contains a symbolic link: {item.filename}")
        names = set(archive.namelist())
        for name in names:
            path = _safe_archive_path(name)
            if not path.parts or path.parts[0] not in {"agent_libos", dist_info}:
                raise ValueError(f"wheel contains a non-core top-level path: {name}")
        missing = sorted(WHEEL_REQUIRED_FILES - names)
        if missing:
            raise ValueError(f"wheel is missing core files: {missing}")
        builtin_paths = {
            name
            for name in names
            if name.startswith("agent_libos/skills/builtin/")
            and name.endswith("/SKILL.md")
        }
        _validate_builtin_skill_archive_payloads(
            {path: archive.read(path) for path in builtin_paths}
        )
        metadata_path = f"{dist_info}/METADATA"
        entry_points_path = f"{dist_info}/entry_points.txt"
        license_path = f"{dist_info}/licenses/LICENSE"
        for required in (metadata_path, entry_points_path, license_path):
            if required not in names:
                raise ValueError(f"wheel is missing {required}")
        metadata = BytesParser(policy=policy.default).parsebytes(archive.read(metadata_path))
        if metadata["Name"] != PROJECT_NAME:
            raise ValueError(f"wheel project name is {metadata['Name']!r}, expected {PROJECT_NAME!r}")
        if metadata["Version"] != version:
            raise ValueError(f"wheel version is {metadata['Version']!r}, expected {version!r}")
        if metadata["Requires-Python"] != "<3.15,>=3.11":
            raise ValueError(
                "wheel Requires-Python must remain >=3.11,<3.15"
            )
        entry_points = archive.read(entry_points_path).decode("utf-8")
        expected_entries = {
            "agent-libos = agent_libos.api.cli:cli",
            "agent-libos-gui-server = agent_libos.api.gui.server:main",
        }
        missing_entries = sorted(entry for entry in expected_entries if entry not in entry_points)
        if missing_entries:
            raise ValueError(f"wheel is missing console entry points: {missing_entries}")


def _looks_like_secret_file(path: PurePosixPath) -> bool:
    return path.name == ".env" or path.suffix.lower() in {".key", ".p12", ".pem", ".pfx"} or (
        any(part.lower() in {"secret", "secrets"} for part in path.parts)
        and "token" in path.name.lower()
    )


def _validate_sdist(sdist_path: Path, version: str) -> None:
    prefix = f"{ARCHIVE_NAME}-{version}"
    relative_files: set[str] = set()
    with tarfile.open(sdist_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            path = _safe_archive_path(member.name)
            if not path.parts or path.parts[0] != prefix:
                raise ValueError(f"sdist path is outside {prefix}: {member.name}")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"sdist contains a non-regular entry: {member.name}")
            relative = PurePosixPath(*path.parts[1:])
            relative_text = relative.as_posix()
            relative_files.add(relative_text)
            if SDIST_FORBIDDEN_PARTS.intersection(relative.parts):
                raise ValueError(f"sdist contains generated or private path: {relative_text}")
            if relative.suffix.lower() in SDIST_FORBIDDEN_SUFFIXES:
                raise ValueError(f"sdist contains generated file: {relative_text}")
            if _looks_like_secret_file(relative) and relative_text not in ALLOWED_SECRET_FIXTURES:
                raise ValueError(f"sdist contains an undeclared secret-like file: {relative_text}")
        metadata_file = archive.extractfile(f"{prefix}/PKG-INFO")
        if metadata_file is None:
            raise ValueError("sdist PKG-INFO cannot be read")
        metadata = BytesParser(policy=policy.default).parsebytes(metadata_file.read())
        if metadata["Name"] != PROJECT_NAME or metadata["Version"] != version:
            raise ValueError("sdist PKG-INFO does not match the release name and version")
        if metadata["Requires-Python"] != "<3.15,>=3.11":
            raise ValueError(
                "sdist Requires-Python must remain >=3.11,<3.15"
            )
        builtin_paths = {
            path
            for path in relative_files
            if path.startswith("agent_libos/skills/builtin/")
            and path.endswith("/SKILL.md")
        }
        builtin_payloads: dict[str, bytes] = {}
        for path in builtin_paths:
            member = archive.extractfile(f"{prefix}/{path}")
            if member is None:
                raise ValueError(f"sdist built-in Skill cannot be read: {path}")
            builtin_payloads[path] = member.read()
        _validate_builtin_skill_archive_payloads(builtin_payloads)
    missing = sorted(SDIST_REQUIRED_FILES - relative_files)
    if missing:
        raise ValueError(f"sdist is missing repository release files: {missing}")


def validate_artifacts(artifact_dir: Path, *, root: Path = ROOT) -> tuple[Path, Path, str]:
    version = validate_version_alignment(root)
    wheel = _single_artifact(artifact_dir, f"{ARCHIVE_NAME}-{version}-*.whl", "wheel")
    sdist = _single_artifact(artifact_dir, f"{ARCHIVE_NAME}-{version}.tar.gz", "sdist")
    _validate_wheel(wheel, version)
    _validate_sdist(sdist, version)
    return wheel, sdist, version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Agent libOS release versions and built artifacts.")
    parser.add_argument("artifact_dir", nargs="?", type=Path, help="Directory containing one wheel and one sdist.")
    parser.add_argument(
        "--version-only",
        action="store_true",
        help="Check only source version alignment; artifact_dir must be omitted.",
    )
    args = parser.parse_args(argv)
    if args.version_only:
        if args.artifact_dir is not None:
            parser.error("artifact_dir cannot be used with --version-only")
        version = validate_version_alignment()
        print(f"release version identifiers are aligned at {version}")
        return 0
    if args.artifact_dir is None:
        parser.error("artifact_dir is required unless --version-only is used")
    wheel, sdist, version = validate_artifacts(args.artifact_dir.resolve())
    print(f"validated {PROJECT_NAME} {version} artifacts: {wheel.name}, {sdist.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
