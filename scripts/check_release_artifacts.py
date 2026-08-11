from __future__ import annotations

import argparse
import ast
from configparser import ConfigParser, Error as ConfigParserError
from email import policy
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import tomllib
import zipfile

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[1]
PROJECT_NAME = "agent-libos"
ARCHIVE_NAME = "agent_libos"
RELEASE_TARGET_VERSION = "1.5.0"
MCP_EXTRA_REQUIREMENTS = (
    "anyio<5,>=4.10; extra == 'mcp'",
    "httpcore2<3,>=2.5; extra == 'mcp'",
    "httpx2<3,>=2.5; extra == 'mcp'",
    "keyring==25.7.0; extra == 'mcp'",
    "mcp==2.0.0; extra == 'mcp'",
    "opentelemetry-api<2,>=1.28; extra == 'mcp'",
)
EXPECTED_CONSOLE_SCRIPTS = {
    "agent-libos": "agent_libos.api.cli:cli",
    "agent-libos-gui-server": "agent_libos.api.gui.server:main",
    "agent-libos-migrate-tool-groups": (
        "agent_libos.storage.tool_skill_migration:cli"
    ),
}
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
_BUILTIN_SKILL_TOOL_COUNT = 101
_BUILTIN_SKILL_FRONTMATTER_FIELDS = frozenset(
    {"name", "description", "allowed-tools"}
)
_BUILTIN_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]*$")


MCP_WHEEL_REQUIRED_FILES = frozenset(
    {
        "agent_libos/mcp/__init__.py",
        "agent_libos/mcp/_input.py",
        "agent_libos/mcp/app_policy.py",
        "agent_libos/mcp/client.py",
        "agent_libos/mcp/continuations.py",
        "agent_libos/mcp/dx.py",
        "agent_libos/mcp/environment.py",
        "agent_libos/mcp/human.py",
        "agent_libos/mcp/manifest.py",
        "agent_libos/mcp/oauth.py",
        "agent_libos/mcp/prompts.py",
        "agent_libos/mcp/providers.py",
        "agent_libos/mcp/resources.py",
        "agent_libos/mcp/runtime_bridge.py",
        "agent_libos/mcp/sdk_subscriptions.py",
        "agent_libos/mcp/side_effects.py",
        "agent_libos/mcp/subscriptions.py",
        "agent_libos/mcp/supervisor.py",
        "agent_libos/mcp/tasks.py",
        "agent_libos/mcp/types.py",
        "agent_libos/mcp/wire.py",
        "agent_libos/storage/mcp_v7.py",
        "agent_libos/storage/mcp_v7_migration.py",
        "agent_libos/storage/postgres_schema_contract.py",
        "agent_libos/storage/postgres_schema_manifest.json",
        "agent_libos/storage/v7_schema_contract.py",
    }
)
MCP_SDIST_REQUIRED_FILES = frozenset(
    {
        "examples/mcp/README.md",
        "examples/mcp/http-v3.yaml",
        "examples/mcp/http_server.py",
        "examples/mcp/run_lifecycle_e2e.py",
        "examples/mcp/run_modern_contract_e2e.py",
        "examples/mcp/run_oauth_e2e.py",
        "examples/mcp/run_probe_scaffold_e2e.py",
        "examples/mcp/run_tools_e2e.py",
        "examples/mcp/stdio-v3.yaml",
        "examples/mcp/stdio_server.py",
        "docs/mcp.md",
        "scripts/check_mcp_test_closure.py",
        "scripts/mcp_conformance_oauth_harness.mts",
        "scripts/mcp_dx.py",
        "scripts/mcp_test_support.py",
        "scripts/run_mcp_conformance.py",
        "scripts/smoke_mcp_extra.py",
        "scripts/smoke_mcp_installed_mrtr_tasks.py",
        "tests/fixtures/mcp_sdk_v2/oauth_tls_server.py",
        "tests/fixtures/mcp_sdk_v2/python_server.py",
        "tests/fixtures/mcp_sdk_v2/tasks_extension_schema.json",
        "tests/fixtures/mcp_sdk_v2/typescript_server/package-lock.json",
        "tests/fixtures/mcp_sdk_v2/typescript_server/package.json",
        "tests/fixtures/mcp_sdk_v2/typescript_server/server.mjs",
    }
)
WHEEL_REQUIRED_FILES = frozenset(
    {
        "agent_libos/__init__.py",
        "agent_libos/__main__.py",
        "agent_libos/api/cli.py",
        "agent_libos/api/gui/server.py",
        "agent_libos/storage/tool_skill_migration.py",
    }
) | BUILTIN_SKILL_ARCHIVE_PATHS | MCP_WHEEL_REQUIRED_FILES
SDIST_REQUIRED_FILES = frozenset(
    {
        "LICENSE",
        "PKG-INFO",
        "README.md",
        "pyproject.toml",
        "agent_libos/__init__.py",
        "agent_libos/storage/tool_skill_migration.py",
        "config.yaml",
        "modules/pty/module.yaml",
        "modules/pty/pty_module.py",
        "skills/swe-agent/SKILL.md",
        "images/mini-swe-agent/IMAGE.yaml",
        "docs/release_status.md",
        "tests/invariants.yaml",
        "scripts/check_release_artifacts.py",
    }
) | WHEEL_REQUIRED_FILES | MCP_SDIST_REQUIRED_FILES
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
ALLOWED_SECRET_FIXTURE_SHA256 = {
    "benchmarks/runtime_safety/fixtures/basic_repo/.env": (
        "8aac0c24c8d27c785c264368e073b15bec29709f0cb340defb6f1c175d772b26"
    ),
    "benchmarks/runtime_safety/fixtures/basic_repo/config/private.key": (
        "d9269c575547491851aaea7f0da8b5b10cc752bea917f4cab10a329d0ab95c96"
    ),
    "benchmarks/runtime_safety/fixtures/basic_repo/secrets/token.txt": (
        "78c8c6a65f94cedb182f5e37ea8ab2df82b89febb1a213df527008870a592777"
    ),
}
ALLOWED_SECRET_FIXTURES = frozenset(ALLOWED_SECRET_FIXTURE_SHA256)
CHECKSUM_MANIFEST_NAME = "SHA256SUMS"
_FINAL_RELEASE_VERSION = re.compile(
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\Z"
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


def _editable_lock_version(
    root: Path,
    relative_path: str,
    *,
    editable: str,
) -> str:
    lock = tomllib.loads((root / relative_path).read_text(encoding="utf-8"))
    matches = [
        package
        for package in lock.get("package", [])
        if package.get("name") == PROJECT_NAME
        and isinstance(package.get("source"), dict)
        and package["source"].get("editable") == editable
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{relative_path} must contain exactly one editable "
            f"{PROJECT_NAME} package from {editable!r}"
        )
    version = matches[0].get("version")
    if not isinstance(version, str):
        raise ValueError(
            f"{relative_path} editable {PROJECT_NAME} version is invalid"
        )
    return version


def _workflow_artifact_version(
    root: Path,
    variable: str,
    *,
    suffix: str,
) -> str:
    workflow_path = root / ".github" / "workflows" / "test.yml"
    values: list[str] = []
    for line in workflow_path.read_text(encoding="utf-8").splitlines():
        key, separator, raw_value = line.strip().partition(":")
        if separator and key == variable:
            values.append(raw_value.strip())
    if len(values) != 1:
        raise ValueError(f"workflow must define {variable} exactly once")
    value = values[0]
    if value[:1] in {'"', "'"}:
        if len(value) < 2 or value[-1] != value[0]:
            raise ValueError(f"workflow {variable} must use balanced quotes")
        value = value[1:-1]
    prefix = f"dist/{ARCHIVE_NAME}-"
    if not value.startswith(prefix) or not value.endswith(suffix):
        raise ValueError(f"workflow {variable} is not a canonical artifact path")
    version = value[len(prefix) : -len(suffix)]
    if not version or "/" in version or "\\" in version:
        raise ValueError(f"workflow {variable} version is invalid")
    return version


def _swe_agent_compatibility_version(root: Path) -> str:
    path = root / "skills" / "swe-agent" / "SKILL.md"
    matches = re.findall(
        r"^compatibility:[ \t]+agent-libos==([^ \t\r\n]+)[ \t]*$",
        path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if len(matches) != 1:
        raise ValueError(
            "skills/swe-agent/SKILL.md must contain exactly one exact "
            "agent-libos compatibility pin"
        )
    return matches[0]


def release_versions(root: Path = ROOT) -> dict[str, str]:
    pyproject = tomllib.loads(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    versions = {
        "pyproject.toml": str(pyproject["project"]["version"]),
        "agent_libos/__init__.py": _python_package_version(root),
        "uv.lock": _editable_lock_version(root, "uv.lock", editable="."),
        "experiments/agentdojo/uv.lock": _editable_lock_version(
            root,
            "experiments/agentdojo/uv.lock",
            editable="../../",
        ),
        ".github/workflows/test.yml RELEASE_WHEEL": _workflow_artifact_version(
            root,
            "RELEASE_WHEEL",
            suffix="-py3-none-any.whl",
        ),
        ".github/workflows/test.yml RELEASE_SDIST": _workflow_artifact_version(
            root,
            "RELEASE_SDIST",
            suffix=".tar.gz",
        ),
        "skills/swe-agent/SKILL.md": _swe_agent_compatibility_version(root),
    }
    gui_package_path = root / "gui" / "package.json"
    gui_lock_path = root / "gui" / "package-lock.json"
    if gui_package_path.exists() != gui_lock_path.exists():
        raise ValueError(
            "GUI package metadata must include both package.json and package-lock.json"
        )
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
    mismatches = {
        source: version
        for source, version in versions.items()
        if version != selected
    }
    if mismatches:
        details = ", ".join(
            f"{source}={version}" for source, version in mismatches.items()
        )
        raise ValueError(
            f"release version identifiers do not match {selected}: {details}"
        )
    if _FINAL_RELEASE_VERSION.fullmatch(selected) is None:
        raise ValueError(
            "release version must use final-form numeric X.Y.Z with ASCII digits "
            "and no leading zeros"
        )
    if selected != RELEASE_TARGET_VERSION:
        raise ValueError(
            f"release target version must be exactly {RELEASE_TARGET_VERSION}, "
            f"found {selected}"
        )
    return selected


def _safe_archive_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"archive contains unsafe path: {name}")
    return path


def _reject_duplicate_archive_paths(names: list[str], *, kind: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in names:
        canonical = _safe_archive_path(name).as_posix()
        if canonical in seen:
            duplicates.add(canonical)
        seen.add(canonical)
    if duplicates:
        raise ValueError(f"{kind} contains duplicate member paths: {sorted(duplicates)}")


def _require_regular_artifact(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"release artifact cannot be inspected: {path.name}") from exc
    if stat.S_ISLNK(mode):
        raise ValueError(f"release artifact must not be a symbolic link: {path.name}")
    if not stat.S_ISREG(mode):
        raise ValueError(f"release artifact must be a regular file: {path.name}")


def _release_artifact_paths(
    artifact_dir: Path,
    version: str,
    *,
    include_checksum_manifest: bool,
) -> tuple[Path, Path, Path | None]:
    try:
        directory_mode = artifact_dir.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"release artifact directory cannot be inspected: {artifact_dir}") from exc
    if stat.S_ISLNK(directory_mode) or not stat.S_ISDIR(directory_mode):
        raise ValueError("release artifact directory must be a real directory")

    entries = sorted(artifact_dir.iterdir(), key=lambda path: path.name)
    for entry in entries:
        _require_regular_artifact(entry)

    wheel_name = f"{ARCHIVE_NAME}-{version}-py3-none-any.whl"
    sdist_name = f"{ARCHIVE_NAME}-{version}.tar.gz"
    expected_names = {wheel_name, sdist_name}
    if include_checksum_manifest:
        expected_names.add(CHECKSUM_MANIFEST_NAME)
    actual_names = {entry.name for entry in entries}
    if actual_names != expected_names:
        raise ValueError(
            "release artifact directory must contain exactly the canonical files: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    manifest = artifact_dir / CHECKSUM_MANIFEST_NAME if include_checksum_manifest else None
    return artifact_dir / wheel_name, artifact_dir / sdist_name, manifest


def _file_sha256(path: Path) -> str:
    _require_regular_artifact(path)
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checksum_manifest_bytes(wheel: Path, sdist: Path) -> bytes:
    return (
        f"{_file_sha256(wheel)}  {wheel.name}\n"
        f"{_file_sha256(sdist)}  {sdist.name}\n"
    ).encode("ascii")


def _verify_checksum_manifest(manifest: Path, wheel: Path, sdist: Path) -> None:
    _require_regular_artifact(manifest)
    try:
        actual = manifest.read_bytes()
    except OSError as exc:
        raise ValueError("release checksum manifest cannot be read") from exc
    expected = _checksum_manifest_bytes(wheel, sdist)
    if actual != expected:
        raise ValueError(
            "release checksum manifest must contain the exact canonical artifact digests"
        )


def write_checksum_manifest(
    artifact_dir: Path,
    wheel: Path,
    sdist: Path,
    version: str,
) -> Path:
    manifest = artifact_dir / CHECKSUM_MANIFEST_NAME
    try:
        with manifest.open("xb") as output:
            output.write(_checksum_manifest_bytes(wheel, sdist))
    except FileExistsError as exc:
        raise ValueError("release checksum manifest already exists") from exc
    except OSError as exc:
        raise ValueError("release checksum manifest cannot be written") from exc
    canonical_wheel, canonical_sdist, canonical_manifest = _release_artifact_paths(
        artifact_dir,
        version,
        include_checksum_manifest=True,
    )
    assert canonical_manifest is not None
    if canonical_wheel != wheel or canonical_sdist != sdist:
        raise ValueError("release artifact paths changed while writing checksums")
    _verify_checksum_manifest(canonical_manifest, canonical_wheel, canonical_sdist)
    return canonical_manifest


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
    if len(owner_by_tool) != _BUILTIN_SKILL_TOOL_COUNT:
        raise ValueError(
            "archive built-in Skill catalog must own exactly "
            f"{_BUILTIN_SKILL_TOOL_COUNT} tools, found {len(owner_by_tool)}"
        )


def _validate_console_scripts(raw: bytes) -> None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("wheel entry_points.txt is not UTF-8") from exc
    parser = ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str
    try:
        parser.read_string(text)
    except ConfigParserError as exc:
        raise ValueError("wheel entry_points.txt is invalid") from exc
    actual = (
        dict(parser.items("console_scripts", raw=True))
        if parser.has_section("console_scripts")
        else {}
    )
    if actual != EXPECTED_CONSOLE_SCRIPTS:
        missing = sorted(EXPECTED_CONSOLE_SCRIPTS.keys() - actual.keys())
        unexpected = sorted(actual.keys() - EXPECTED_CONSOLE_SCRIPTS.keys())
        mismatched = sorted(
            name
            for name in EXPECTED_CONSOLE_SCRIPTS.keys() & actual.keys()
            if actual[name] != EXPECTED_CONSOLE_SCRIPTS[name]
        )
        raise ValueError(
            "wheel console entry points mismatch: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )


def _requirement_signature(requirement: Requirement) -> tuple[str, tuple[str, ...], str, str]:
    return (
        canonicalize_name(requirement.name),
        tuple(sorted(canonicalize_name(extra) for extra in requirement.extras)),
        str(requirement.specifier),
        str(requirement.marker or ""),
    )


def _is_mcp_extra_requirement(requirement: Requirement) -> bool:
    marker = str(requirement.marker or "")
    return bool(
        re.search(r'\bextra\s*==\s*["\']mcp["\']', marker)
        or re.search(r'["\']mcp["\']\s*==\s*extra\b', marker)
    )


def _validate_mcp_extra_metadata(metadata: object, *, artifact: str) -> None:
    get_all = getattr(metadata, "get_all", None)
    if not callable(get_all):
        raise ValueError(f"{artifact} metadata cannot be inspected")

    extras = [canonicalize_name(value) for value in get_all("Provides-Extra", [])]
    if extras.count("mcp") != 1:
        raise ValueError(
            f"{artifact} metadata must declare Provides-Extra: mcp exactly once"
        )

    parsed: list[Requirement] = []
    for value in get_all("Requires-Dist", []):
        try:
            parsed.append(Requirement(value))
        except InvalidRequirement as exc:
            raise ValueError(
                f"{artifact} metadata contains an invalid Requires-Dist: {value!r}"
            ) from exc
    actual = [
        _requirement_signature(requirement)
        for requirement in parsed
        if _is_mcp_extra_requirement(requirement)
    ]
    expected = [
        _requirement_signature(Requirement(value))
        for value in MCP_EXTRA_REQUIREMENTS
    ]
    if sorted(actual) != sorted(expected):
        raise ValueError(
            f"{artifact} metadata MCP extra requirements mismatch: "
            f"expected={sorted(expected)!r}, actual={sorted(actual)!r}"
        )


def _validate_wheel(wheel_path: Path, version: str) -> None:
    dist_info = f"{ARCHIVE_NAME}-{version}.dist-info"
    expected_wheel_name = f"{ARCHIVE_NAME}-{version}-py3-none-any.whl"
    if wheel_path.name != expected_wheel_name:
        raise ValueError(
            "wheel filename must use the exact py3-none-any tag: "
            f"{wheel_path.name} != {expected_wheel_name}"
        )
    with zipfile.ZipFile(wheel_path) as archive:
        _reject_duplicate_archive_paths(archive.namelist(), kind="wheel")
        for item in archive.infolist():
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError(f"wheel contains a symbolic link: {item.filename}")
        names = set(archive.namelist())
        for name in names:
            path = _safe_archive_path(name)
            if not path.parts or path.parts[0] not in {"agent_libos", dist_info}:
                raise ValueError(f"wheel contains a non-core top-level path: {name}")
            if _looks_like_secret_file(path):
                raise ValueError(
                    "wheel contains a secret-like file: "
                    f"{path.as_posix()}"
                )
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
        wheel_metadata_path = f"{dist_info}/WHEEL"
        entry_points_path = f"{dist_info}/entry_points.txt"
        license_path = f"{dist_info}/licenses/LICENSE"
        for required in (
            metadata_path,
            wheel_metadata_path,
            entry_points_path,
            license_path,
        ):
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
        _validate_mcp_extra_metadata(metadata, artifact="wheel")
        wheel_metadata = BytesParser(policy=policy.default).parsebytes(
            archive.read(wheel_metadata_path)
        )
        purelib_values = wheel_metadata.get_all("Root-Is-Purelib", [])
        if purelib_values != ["true"]:
            raise ValueError(
                "wheel WHEEL metadata must contain exactly "
                "Root-Is-Purelib: true"
            )
        tags = wheel_metadata.get_all("Tag", [])
        if tags != ["py3-none-any"]:
            raise ValueError(
                "wheel WHEEL metadata must contain exactly one Tag: "
                "py3-none-any"
            )
        _validate_console_scripts(archive.read(entry_points_path))


def _looks_like_secret_file(path: PurePosixPath) -> bool:
    folded_name = path.name.casefold()
    return folded_name == ".env" or folded_name.startswith(".env.") or path.suffix.lower() in {".key", ".p12", ".pem", ".pfx"} or (
        any(part.lower() in {"secret", "secrets"} for part in path.parts)
        and "token" in folded_name
    )


def _validate_sdist(sdist_path: Path, version: str) -> None:
    prefix = f"{ARCHIVE_NAME}-{version}"
    relative_files: set[str] = set()
    with tarfile.open(sdist_path, mode="r:gz") as archive:
        members = archive.getmembers()
        _reject_duplicate_archive_paths(
            [member.name for member in members],
            kind="sdist",
        )
        for member in members:
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
            if _looks_like_secret_file(relative):
                expected_digest = ALLOWED_SECRET_FIXTURE_SHA256.get(relative_text)
                if expected_digest is None:
                    raise ValueError(
                        "sdist contains an undeclared secret-like file: "
                        f"{relative_text}"
                    )
                fixture_file = archive.extractfile(member)
                if fixture_file is None:
                    raise ValueError(
                        f"sdist secret-like fixture cannot be read: {relative_text}"
                    )
                actual_digest = hashlib.sha256(fixture_file.read()).hexdigest()
                if actual_digest != expected_digest:
                    raise ValueError(
                        "sdist allowed secret-like fixture content mismatch: "
                        f"{relative_text}"
                    )
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
        _validate_mcp_extra_metadata(metadata, artifact="sdist")
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


def validate_artifacts(
    artifact_dir: Path,
    *,
    root: Path = ROOT,
    verify_checksums: bool = False,
) -> tuple[Path, Path, str]:
    version = validate_version_alignment(root)
    wheel, sdist, manifest = _release_artifact_paths(
        artifact_dir,
        version,
        include_checksum_manifest=verify_checksums,
    )
    _validate_wheel(wheel, version)
    _validate_sdist(sdist, version)
    if verify_checksums:
        assert manifest is not None
        _verify_checksum_manifest(manifest, wheel, sdist)
    return wheel, sdist, version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Agent libOS release versions and built artifacts.")
    parser.add_argument("artifact_dir", nargs="?", type=Path, help="Directory containing one wheel and one sdist.")
    parser.add_argument(
        "--version-only",
        action="store_true",
        help="Check only source version alignment; artifact_dir must be omitted.",
    )
    checksum_group = parser.add_mutually_exclusive_group()
    checksum_group.add_argument(
        "--write-checksums",
        action="store_true",
        help="Write an exact SHA256SUMS manifest after validating the two artifacts.",
    )
    checksum_group.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Require and verify the exact SHA256SUMS manifest.",
    )
    args = parser.parse_args(argv)
    if args.version_only:
        if args.artifact_dir is not None:
            parser.error("artifact_dir cannot be used with --version-only")
        if args.write_checksums or args.verify_checksums:
            parser.error("checksum options cannot be used with --version-only")
        version = validate_version_alignment()
        print(f"release version identifiers are aligned at {version}")
        return 0
    if args.artifact_dir is None:
        parser.error("artifact_dir is required unless --version-only is used")
    wheel, sdist, version = validate_artifacts(
        args.artifact_dir,
        verify_checksums=args.verify_checksums,
    )
    if args.write_checksums:
        write_checksum_manifest(args.artifact_dir, wheel, sdist, version)
    print(f"validated {PROJECT_NAME} {version} artifacts: {wheel.name}, {sdist.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
