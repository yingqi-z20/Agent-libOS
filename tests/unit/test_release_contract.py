from __future__ import annotations

import ast
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile

import pytest
import yaml

from agent_libos import __version__
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import SEMANTIC_STATUS_SCHEMA_VERSION
from agent_libos.skills.builtin_catalog import (
    BUILTIN_SKILL_IDS as RUNTIME_BUILTIN_SKILL_IDS,
    BUILTIN_SKILL_MAX_FILE_BYTES as RUNTIME_BUILTIN_SKILL_MAX_FILE_BYTES,
    BUILTIN_SKILL_MAX_INSTRUCTION_BYTES as RUNTIME_BUILTIN_SKILL_MAX_INSTRUCTION_BYTES,
)
from agent_libos.storage import STORE_SCHEMA_VERSION
from scripts.check_release_artifacts import (
    ALLOWED_SECRET_FIXTURE_SHA256,
    BUILTIN_SKILL_ARCHIVE_PATHS,
    BUILTIN_SKILL_IDS,
    CHECKSUM_MANIFEST_NAME,
    EXPECTED_CONSOLE_SCRIPTS,
    MCP_EXTRA_REQUIREMENTS,
    MCP_SDIST_REQUIRED_FILES,
    MCP_WHEEL_REQUIRED_FILES,
    SDIST_REQUIRED_FILES,
    WHEEL_REQUIRED_FILES,
    _BUILTIN_SKILL_MAX_FILE_BYTES,
    _BUILTIN_SKILL_MAX_INSTRUCTION_BYTES,
    _validate_builtin_skill_archive_payloads,
    _validate_console_scripts,
    _validate_sdist,
    _validate_wheel,
    validate_artifacts,
    validate_version_alignment,
    write_checksum_manifest,
)
from tests.conftest import pytest_sessionfinish


ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_BLOB_ROOT = "https://github.com/yingqi-z20/Agent-libOS/blob/main/"
ACTION_PINS = {
    "actions/checkout": (
        "3d3c42e5aac5ba805825da76410c181273ba90b1",
        "v7.0.1",
    ),
    "actions/setup-python": (
        "5fda3b95a4ea91299a34e894583c3862153e4b97",
        "v7.0.0",
    ),
    "actions/setup-node": (
        "820762786026740c76f36085b0efc47a31fe5020",
        "v7.0.0",
    ),
    "astral-sh/setup-uv": (
        "c771a70e6277c0a99b617c7a806ffedaca235ff9",
        "v9.0.0",
    ),
    "denoland/setup-deno": (
        "22d081ff2d3a40755e97629de92e3bcbfa7cf2ed",
        "v2.0.5",
    ),
    "actions/upload-artifact": (
        "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
        "v7.0.1",
    ),
    "actions/download-artifact": (
        "3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
        "v8.0.1",
    ),
}


def _wheel_metadata(
    version: str,
    *,
    include_mcp_extra: bool = True,
    mcp_requirements: tuple[str, ...] = MCP_EXTRA_REQUIREMENTS,
) -> bytes:
    metadata = (
        "Metadata-Version: 2.4\n"
        "Name: agent-libos\n"
        f"Version: {version}\n"
        "Requires-Python: <3.15,>=3.11\n"
    )
    if include_mcp_extra:
        metadata += "Provides-Extra: mcp\n"
    metadata += "".join(
        f"Requires-Dist: {requirement}\n" for requirement in mcp_requirements
    )
    return f"{metadata}\n".encode()


def _write_test_wheel(
    target: Path,
    *,
    version: str = "1.0.0",
    wheel_metadata: bytes | None = None,
    project_metadata: bytes | None = None,
) -> Path:
    dist_info = f"agent_libos-{version}.dist-info"
    console_scripts = "\n".join(
        ["[console_scripts]"]
        + [
            f"{name} = {entrypoint}"
            for name, entrypoint in EXPECTED_CONSOLE_SCRIPTS.items()
        ]
    ).encode()
    with zipfile.ZipFile(target, "w") as archive:
        for relative in WHEEL_REQUIRED_FILES:
            archive.writestr(relative, (ROOT / relative).read_bytes())
        archive.writestr(
            f"{dist_info}/METADATA",
            project_metadata or _wheel_metadata(version),
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            wheel_metadata
            or (
                b"Wheel-Version: 1.0\n"
                b"Generator: release-contract-test\n"
                b"Root-Is-Purelib: true\n"
                b"Tag: py3-none-any\n"
                b"\n"
            ),
        )
        archive.writestr(f"{dist_info}/entry_points.txt", console_scripts)
        archive.writestr(f"{dist_info}/licenses/LICENSE", b"test license\n")
    return target


def _write_test_sdist(
    target: Path,
    *,
    duplicate_path: str | None = None,
    secret_overrides: dict[str, bytes] | None = None,
    version: str = "1.0.0",
    project_metadata: bytes | None = None,
) -> Path:
    prefix = f"agent_libos-{version}"
    payloads: dict[str, bytes] = {}
    for relative in SDIST_REQUIRED_FILES:
        if relative == "PKG-INFO":
            payloads[relative] = project_metadata or _wheel_metadata(version)
        elif relative in BUILTIN_SKILL_ARCHIVE_PATHS:
            payloads[relative] = (ROOT / relative).read_bytes()
        else:
            payloads[relative] = b"release contract fixture\n"
    for relative in ALLOWED_SECRET_FIXTURE_SHA256:
        payloads[relative] = (ROOT / relative).read_bytes()
    payloads.update(secret_overrides or {})

    with tarfile.open(target, "w:gz") as archive:
        for relative, raw in sorted(payloads.items()):
            info = tarfile.TarInfo(f"{prefix}/{relative}")
            info.size = len(raw)
            info.mode = 0o644
            archive.addfile(info, BytesIO(raw))
        if duplicate_path is not None:
            raw = payloads[duplicate_path]
            info = tarfile.TarInfo(f"{prefix}/{duplicate_path}")
            info.size = len(raw)
            info.mode = 0o644
            archive.addfile(info, BytesIO(raw))
    return target


def _write_release_pair(target: Path, *, version: str = "1.5.0") -> tuple[Path, Path]:
    wheel = _write_test_wheel(
        target / f"agent_libos-{version}-py3-none-any.whl",
        version=version,
    )
    sdist = _write_test_sdist(
        target / f"agent_libos-{version}.tar.gz",
        version=version,
    )
    return wheel, sdist


def test_release_version_identifiers_are_aligned() -> None:
    assert validate_version_alignment(ROOT) == "1.5.0"


def test_release_protocol_versions_are_independent_and_default_off() -> None:
    assert __version__ == "1.5.0"
    assert STORE_SCHEMA_VERSION == 7
    assert SEMANTIC_STATUS_SCHEMA_VERSION == 3
    assert DEFAULT_CONFIG.semantic.mode == "off"
    assert DEFAULT_CONFIG.semantic.policy_epoch is None


def test_agentdojo_lock_tracks_current_editable_agent_libos_metadata() -> None:
    lock = tomllib.loads(
        (ROOT / "experiments" / "agentdojo" / "uv.lock").read_text(
            encoding="utf-8"
        )
    )
    package = next(item for item in lock["package"] if item["name"] == "agent-libos")
    assert package["version"] == "1.5.0"
    assert package["source"] == {"editable": "../../"}
    assert {
        (item["specifier"], item["marker"])
        for item in package["metadata"]["requires-dist"]
        if item["name"] == "pywinpty"
    } == {(">=3.0.5,<4", "sys_platform == 'win32' and extra == 'pty'")}


def test_root_lock_resolves_the_reviewed_mcp_sdk_v2_graph() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    packages = {item["name"]: item for item in lock["package"]}

    assert packages["agent-libos"]["version"] == "1.5.0"
    assert packages["mcp"]["version"] == "2.0.0"
    assert packages["mcp-types"]["version"] == "2.0.0"
    assert {
        item["specifier"]
        for item in packages["agent-libos"]["metadata"]["requires-dist"]
        if item["name"] == "mcp" and item.get("marker") == "extra == 'mcp'"
    } == {"==2.0.0"}
    assert {
        item["specifier"]
        for item in packages["agent-libos"]["metadata"]["requires-dist"]
        if item["name"] == "keyring" and item.get("marker") == "extra == 'mcp'"
    } == {"==25.7.0"}
    assert {
        item["name"]
        for item in packages["agent-libos"]["optional-dependencies"]["mcp"]
    } == {"anyio", "mcp", "httpx2", "httpcore2", "keyring", "opentelemetry-api"}


def test_root_lock_resolves_the_reviewed_windows_pty_graph() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    packages = {item["name"]: item for item in lock["package"]}
    project = packages["agent-libos"]
    pywinpty = packages["pywinpty"]

    assert pywinpty["version"] == "3.0.5"
    assert project["optional-dependencies"]["pty"] == [
        {"name": "pywinpty", "marker": "sys_platform == 'win32'"}
    ]
    assert {
        (item["specifier"], item["marker"])
        for item in project["metadata"]["requires-dist"]
        if item["name"] == "pywinpty"
    } == {(">=3.0.5,<4", "sys_platform == 'win32' and extra == 'pty'")}
    assert any(
        "cp314-cp314-win_amd64.whl" in wheel["url"]
        for wheel in pywinpty["wheels"]
    )


def test_gui_lockfile_uses_only_the_public_npm_registry() -> None:
    lock = json.loads((ROOT / "gui" / "package-lock.json").read_text(encoding="utf-8"))
    resolved_urls = sorted(
        package["resolved"]
        for package in lock["packages"].values()
        if isinstance(package, dict) and isinstance(package.get("resolved"), str)
    )

    assert resolved_urls
    assert [
        url
        for url in resolved_urls
        if not url.startswith("https://registry.npmjs.org/")
    ] == []


def test_tracked_project_files_contain_no_private_vendor_markers() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\0")
    forbidden = (b"bn" + b"pm", b"by" + b"ted", b"byte" + b"dance")
    offenders: list[str] = []

    for relative in tracked:
        if not relative:
            continue
        payload = (ROOT / relative).read_bytes().lower()
        if any(marker in payload for marker in forbidden):
            offenders.append(relative)

    assert offenders == []


def test_build_backend_runtime_dependencies_and_project_urls_are_bounded() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["build-system"] == {
        "requires": ["hatchling==1.31.0"],
        "build-backend": "hatchling.build",
    }
    assert set(pyproject["project"]["dependencies"]) == {
        "openai>=2.43.0,<3",
        "psutil>=7.0.0,<8",
        "pydantic>=2.13.4,<3",
        "jsonschema>=4.25.0,<5",
        "pyyaml>=6.0.3,<7",
        "regex>=2024.11.6,<2027",
    }
    assert pyproject["project"]["optional-dependencies"] == {
        "postgres": ["psycopg[binary]>=3.2,<4"],
        "mcp": [
            "anyio>=4.10,<5",
            "mcp==2.0.0",
            "httpx2>=2.5,<3",
            "httpcore2>=2.5,<3",
            "keyring==25.7.0",
            "opentelemetry-api>=1.28,<2",
        ],
        "pty": ["pywinpty>=3.0.5,<4; sys_platform == 'win32'"],
    }
    assert pyproject["dependency-groups"]["release"] == [
        "check-wheel-contents==0.6.3",
        "hatchling==1.31.0",
        "packaging==26.2",
        "twine==7.0.0",
    ]
    assert pyproject["project"]["urls"] == {
        "Homepage": "https://github.com/yingqi-z20/Agent-libOS",
        "Documentation": f"{REPOSITORY_BLOB_ROOT}README.md#documentation",
        "Repository": "https://github.com/yingqi-z20/Agent-libOS",
        "Issues": "https://github.com/yingqi-z20/Agent-libOS/issues",
        "Release status": f"{REPOSITORY_BLOB_ROOT}docs/release_status.md",
    }


def test_readme_has_no_relative_repository_links_in_pypi_metadata() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    destinations = re.findall(r"\[[^\]]+\]\(([^)]+)\)", readme)
    relative = [
        destination
        for destination in destinations
        if not destination.startswith(("https://", "http://", "mailto:", "#"))
    ]

    assert relative == []
    repository_destinations = {
        destination
        for destination in destinations
        if destination.startswith(REPOSITORY_BLOB_ROOT)
    }
    missing_targets = [
        destination
        for destination in sorted(repository_destinations)
        if not (
            ROOT
            / destination.removeprefix(REPOSITORY_BLOB_ROOT).split("#", 1)[0]
        ).is_file()
    ]
    assert missing_targets == []
    assert {
        f"{REPOSITORY_BLOB_ROOT}docs/tools_and_jit.md#on-demand-tool-skills",
        f"{REPOSITORY_BLOB_ROOT}docs/storage.md#transaction-model",
        (
            f"{REPOSITORY_BLOB_ROOT}docs/configuration.md"
            "#effective-llm-profile-precedence"
        ),
        f"{REPOSITORY_BLOB_ROOT}docs/development.md#real-llm-smoke",
    } <= repository_destinations


@pytest.mark.parametrize(
    ("filename", "wheel_metadata", "error"),
    (
        (
            "agent_libos-1.0.0-py3-none-any.whl",
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: py3-none-any\n\n",
            "Root-Is-Purelib",
        ),
        (
            "agent_libos-1.0.0-py3-none-any.whl",
            (
                b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\n"
                b"Tag: py3-none-any\nTag: cp311-cp311-macosx_11_0_arm64\n\n"
            ),
            "exactly one Tag",
        ),
        (
            "agent_libos-1.0.0-py3-none-macosx_11_0_arm64.whl",
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n",
            "filename",
        ),
    ),
)
def test_wheel_validation_rejects_tampered_pure_python_tags(
    tmp_path: Path,
    filename: str,
    wheel_metadata: bytes,
    error: str,
) -> None:
    wheel = _write_test_wheel(
        tmp_path / filename,
        wheel_metadata=wheel_metadata,
    )

    with pytest.raises(ValueError, match=error):
        _validate_wheel(wheel, "1.0.0")


def test_sdist_validation_rejects_tampered_allowed_secret_fixture(
    tmp_path: Path,
) -> None:
    valid = _write_test_sdist(tmp_path / "valid.tar.gz")
    _validate_sdist(valid, "1.0.0")

    fixture_path = "benchmarks/runtime_safety/fixtures/basic_repo/.env"
    tampered = _write_test_sdist(
        tmp_path / "tampered.tar.gz",
        secret_overrides={fixture_path: b"OPENAI_API_KEY=real-secret\n"},
    )
    with pytest.raises(ValueError, match="secret-like fixture content mismatch"):
        _validate_sdist(tampered, "1.0.0")


def test_wheel_validation_requires_the_mcp_provides_extra_marker(
    tmp_path: Path,
) -> None:
    wheel = _write_test_wheel(
        tmp_path / "agent_libos-1.0.0-py3-none-any.whl",
        project_metadata=_wheel_metadata("1.0.0", include_mcp_extra=False),
    )

    with pytest.raises(ValueError, match="Provides-Extra: mcp exactly once"):
        _validate_wheel(wheel, "1.0.0")


def test_sdist_validation_rejects_missing_or_widened_mcp_dependencies(
    tmp_path: Path,
) -> None:
    missing = _write_test_sdist(
        tmp_path / "missing.tar.gz",
        project_metadata=_wheel_metadata(
            "1.0.0",
            mcp_requirements=MCP_EXTRA_REQUIREMENTS[:-1],
        ),
    )
    widened_requirements = tuple(
        "mcp>=2.0; extra == 'mcp'" if requirement.startswith("mcp==") else requirement
        for requirement in MCP_EXTRA_REQUIREMENTS
    )
    widened = _write_test_sdist(
        tmp_path / "widened.tar.gz",
        project_metadata=_wheel_metadata(
            "1.0.0",
            mcp_requirements=widened_requirements,
        ),
    )

    for artifact in (missing, widened):
        with pytest.raises(ValueError, match="MCP extra requirements mismatch"):
            _validate_sdist(artifact, "1.0.0")


def test_artifact_validation_rejects_widened_default_keyring_dependency(
    tmp_path: Path,
) -> None:
    widened_requirements = tuple(
        "keyring>=25.7,<26; extra == 'mcp'"
        if requirement.startswith("keyring==")
        else requirement
        for requirement in MCP_EXTRA_REQUIREMENTS
    )
    wheel = _write_test_wheel(
        tmp_path / "agent_libos-1.0.0-py3-none-any.whl",
        project_metadata=_wheel_metadata(
            "1.0.0",
            mcp_requirements=widened_requirements,
        ),
    )
    sdist = _write_test_sdist(
        tmp_path / "agent_libos-1.0.0.tar.gz",
        project_metadata=_wheel_metadata(
            "1.0.0",
            mcp_requirements=widened_requirements,
        ),
    )

    with pytest.raises(ValueError, match="MCP extra requirements mismatch"):
        _validate_wheel(wheel, "1.0.0")
    with pytest.raises(ValueError, match="MCP extra requirements mismatch"):
        _validate_sdist(sdist, "1.0.0")


def test_wheel_validation_rejects_duplicate_member_paths(tmp_path: Path) -> None:
    wheel = _write_test_wheel(tmp_path / "agent_libos-1.0.0-py3-none-any.whl")
    with zipfile.ZipFile(wheel, "a") as archive:
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("agent_libos/__init__.py", b"duplicate member")

    with pytest.raises(ValueError, match="duplicate member paths"):
        _validate_wheel(wheel, "1.0.0")


def test_sdist_validation_rejects_duplicate_member_paths(tmp_path: Path) -> None:
    sdist = _write_test_sdist(
        tmp_path / "agent_libos-1.0.0.tar.gz",
        duplicate_path="README.md",
    )

    with pytest.raises(ValueError, match="duplicate member paths"):
        _validate_sdist(sdist, "1.0.0")


@pytest.mark.parametrize("extra_name", ("unexpected.whl", "notes.txt"))
def test_release_artifact_validation_rejects_extra_files(
    tmp_path: Path,
    extra_name: str,
) -> None:
    _write_release_pair(tmp_path)
    (tmp_path / extra_name).write_bytes(b"not a canonical release artifact")

    with pytest.raises(ValueError, match="unexpected"):
        validate_artifacts(tmp_path, root=ROOT)


def test_release_artifact_validation_rejects_non_regular_entries(
    tmp_path: Path,
) -> None:
    _write_release_pair(tmp_path)
    (tmp_path / "unexpected-directory").mkdir()

    with pytest.raises(ValueError, match="regular file"):
        validate_artifacts(tmp_path, root=ROOT)


def test_release_artifact_validation_rejects_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel, _ = _write_release_pair(tmp_path)
    original_lstat = Path.lstat

    def fake_lstat(path: Path):  # type: ignore[no-untyped-def]
        result = original_lstat(path)
        if path == wheel:
            values = list(result)
            values[0] = (values[0] & ~0o170000) | 0o120000
            return type(result)(values)
        return result

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    with pytest.raises(ValueError, match="symbolic link"):
        validate_artifacts(tmp_path, root=ROOT)


def test_release_checksum_manifest_records_and_verifies_exact_artifacts(
    tmp_path: Path,
) -> None:
    wheel, sdist = _write_release_pair(tmp_path)
    validated_wheel, validated_sdist, version = validate_artifacts(
        tmp_path,
        root=ROOT,
    )
    assert (validated_wheel, validated_sdist) == (wheel, sdist)

    manifest = write_checksum_manifest(
        tmp_path,
        validated_wheel,
        validated_sdist,
        version,
    )
    assert manifest.name == CHECKSUM_MANIFEST_NAME
    lines = manifest.read_text(encoding="ascii").splitlines()
    assert [line.split("  ", 1)[1] for line in lines] == [wheel.name, sdist.name]
    validate_artifacts(tmp_path, root=ROOT, verify_checksums=True)

    wheel.write_bytes(wheel.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="exact canonical artifact digests"):
        validate_artifacts(tmp_path, root=ROOT, verify_checksums=True)


def test_release_status_contains_current_version_state_only() -> None:
    text = (ROOT / "docs" / "release_status.md").read_text(encoding="utf-8")
    assert text.startswith("# Agent libOS 1.5.0 Status\n")
    forbidden = {
        "dirty state": r"\bdirty\b",
        "worktree state": r"\bwork(?:ing)?[ -]?tree\b",
        "content hash": r"\bsha(?:-?256)?\b",
        "benchmark artifact path": r"\.benchmark_runs/",
        "absolute user path": r"(?:/Users/|/home/|/private/|/tmp/|[A-Za-z]:\\Users\\)",
        "bare hexadecimal identifier": r"\b[0-9a-f]{7,40}\b",
    }
    offenders = [label for label, pattern in forbidden.items() if re.search(pattern, text, re.IGNORECASE)]
    assert offenders == []
    assert set(re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", text)) <= {
        "2026-07-28"
    }


def test_release_status_requires_an_immutable_ci_receipt_binding() -> None:
    text = (ROOT / "docs" / "release_status.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    for required in (
        "it is not itself a CI receipt",
        "the exact source commit locator",
        "the CI workflow run locator and the required job locators",
        "checksum-manifest artifact locators",
        "not an observed pass for a checkout or candidate artifact",
        "## Validation contract (CI receipt required)",
    ):
        assert required in normalized

    for unbound_claim in (
        "The per-lane deterministic matrix passes",
        "The GUI lane passes",
        "The practical-workflow evaluation passes",
    ):
        assert unbound_claim not in text


def test_semantic_phase2_to_4_release_contract_is_default_off_and_host_controlled() -> None:
    release = " ".join(
        (ROOT / "docs" / "release_status.md").read_text(encoding="utf-8").split()
    )
    semantic = " ".join(
        (ROOT / "docs" / "semantic_shadow.md").read_text(encoding="utf-8").split()
    )
    support = " ".join(
        (ROOT / "docs" / "support_matrix.md").read_text(encoding="utf-8").split()
    )

    for required in (
        "Semantic approval and ingress classification remain default-off",
        "immutable static Host policy epoch",
        "Classifier output is only a veto/escalation signal",
        "never an allow predicate or safety oracle",
        "Semantic HTTP and GUI surfaces remain read-only",
        "no remotely reachable policy activation or revocation endpoint",
    ):
        assert required in release
    for required in (
        "The default remains `semantic.mode: off`",
        "The classifier remains evidence, not authority",
        "The machine path never installs `always_allow`",
        "There is no wildcard tenant, implicit epoch, auto-activation",
        "There is no semantic `POST`, `PUT`, `PATCH`, or `DELETE` route",
    ):
        assert required in semantic
    assert "Default-off Phase 2–4 plane" in support
    assert "Real classifier smoke is opt-in and never a safety oracle" in support
    assert "Shadow-only" not in release
    assert "Shadow-only" not in support


def test_release_status_references_do_not_describe_a_metadata_ledger() -> None:
    documents = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    forbidden = ("commit", "dirty", "worktree", "working-tree", "ledger", "sha-256", "sha256", "exact commands")
    offenders: list[str] = []
    for document in documents:
        lines = document.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if "release_status.md" not in line:
                continue
            context = " ".join(lines[max(0, index - 1) : index + 2]).lower()
            if any(term in context for term in forbidden):
                offenders.append(f"{document.relative_to(ROOT)}:{index + 1}")
    assert offenders == []


def test_release_status_bounds_unarchived_evidence_and_volatile_counts() -> None:
    text = (ROOT / "docs" / "release_status.md").read_text(encoding="utf-8")

    assert "## Unarchived real-LLM observation" in text
    assert "not Agent libOS 1.5.0 release evidence" in text
    assert "AgentDojo harness is a required CI matrix" in text
    assert "collected pytest nodes" not in text
    assert not re.search(r"selects [\d,]+ tests", text)
    for stale_count in ("241,038", "304,779", "138,621"):
        assert stale_count not in text


def test_release_status_gui_evidence_avoids_volatile_suite_counts() -> None:
    text = (ROOT / "docs" / "release_status.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    gui_test_files = {
        path.relative_to(ROOT)
        for source_root in (ROOT / "gui" / "src", ROOT / "gui" / "electron")
        for pattern in ("*.test.ts", "*.test.tsx")
        for path in source_root.rglob(pattern)
    }

    assert gui_test_files
    assert "The GUI job requires the complete checked-in Vitest suite" in normalized
    assert (
        "Exact file and test counts are intentionally left to the bound CI receipt"
        in normalized
    )
    assert not re.search(r"\b\d+ Vitest files\b", text)
    assert not re.search(r"\b\d+ tests\b", text)


def test_documented_no_skip_provider_gates_invoke_pytest_directly() -> None:
    text = (ROOT / "docs" / "development.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert 'tests -m "mcp and not postgres"' in normalized
    assert "tests/fixtures/mcp_sdk_v2/typescript_server" in text
    assert "ci --ignore-scripts --no-audit --no-fund" in normalized
    assert "closure rule" in normalized
    assert "tests/self_evolution/test_builtin_agent_images_real_llm.py" in text
    assert not re.search(r"scripts/test_matrix\.py[^\n]*--fail-on-skip", text)
    assert "`--fail-on-skip` is a pytest option" in normalized


def test_cross_sdk_v2_fixture_is_frozen_and_independent() -> None:
    fixture_root = ROOT / "tests" / "fixtures" / "mcp_sdk_v2"
    package = json.loads(
        (fixture_root / "typescript_server" / "package.json").read_text(
            encoding="utf-8"
        )
    )
    lock = json.loads(
        (fixture_root / "typescript_server" / "package-lock.json").read_text(
            encoding="utf-8"
        )
    )
    server = (fixture_root / "typescript_server" / "server.mjs").read_text(
        encoding="utf-8"
    )
    python_server = (fixture_root / "python_server.py").read_text(encoding="utf-8")

    assert package["private"] is True
    assert package["engines"] == {"node": ">=24"}
    assert package["dependencies"] == {
        "@modelcontextprotocol/server": "2.0.0-beta.4",
        "zod": "4.2.1",
    }
    assert lock["lockfileVersion"] == 3
    assert lock["packages"][""]["dependencies"] == package["dependencies"]
    assert all(
        not entry.get("resolved", "").startswith(("http://", "https://"))
        or entry["resolved"].startswith("https://registry.npmjs.org/")
        for entry in lock["packages"].values()
    )
    for surface in ("registerResource", "registerPrompt", "sendResourceUpdated"):
        assert surface in server
    for surface in ("@server.resource", "@server.prompt", "notify_resource_updated"):
        assert surface in python_server


def test_mcp_sdk_optional_test_files_are_explicitly_marked() -> None:
    expected_marked = (
        "tests/providers/test_mcp_cross_sdk_v2.py",
        "tests/providers/test_mcp_http_transport.py",
        "tests/providers/test_mcp_modern_cli_e2e.py",
        "tests/providers/test_mcp_modern_mrtr_sdk.py",
        "tests/providers/test_mcp_oauth_runtime_tls.py",
        "tests/providers/test_mcp_resources_prompts.py",
        "tests/providers/test_mcp_sdk_subscriptions.py",
        "tests/providers/test_mcp_v2_adapter.py",
        "tests/providers/test_mcp_sdk_integration.py",
    )
    for relative in expected_marked:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "pytest.mark.mcp_transport" in text
    actual_marked = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests" / "providers").glob("test_mcp*.py")
        if "pytest.mark.mcp_transport" in path.read_text(encoding="utf-8")
    }
    assert actual_marked == set(expected_marked)


def test_mcp_sdk_optional_test_files_defer_nondefault_imports() -> None:
    optional_roots = {
        "cryptography",
        "httpcore2",
        "httpx2",
        "keyring",
        "mcp",
        "opentelemetry",
    }
    offenders: list[str] = []

    for path in (ROOT / "tests" / "providers").glob("test_mcp*.py"):
        source = path.read_text(encoding="utf-8")
        if "pytest.mark.mcp_transport" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        for statement in tree.body:
            imported_roots: set[str] = set()
            if isinstance(statement, ast.Import):
                imported_roots = {alias.name.partition(".")[0] for alias in statement.names}
            elif isinstance(statement, ast.ImportFrom) and statement.module:
                imported_roots = {statement.module.partition(".")[0]}
            elif (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Attribute)
                and isinstance(statement.value.func.value, ast.Name)
                and statement.value.func.value.id == "pytest"
                and statement.value.func.attr == "importorskip"
            ):
                offenders.append(path.relative_to(ROOT).as_posix())
                continue
            if imported_roots & optional_roots:
                offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []


def test_mcp_provider_gate_is_closed_over_the_full_pytest_tree() -> None:
    conftest = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    checker = (ROOT / "scripts" / "check_mcp_test_closure.py").read_text(
        encoding="utf-8"
    )
    assert "@pytest.hookimpl(tryfirst=True)" in conftest
    assert "_is_mcp_test_item" in conftest
    assert 'item.add_marker(pytest.mark.mcp)' in conftest
    assert '"mcp_transport" in item.keywords' in conftest
    assert "expected - marked" in checker
    assert "transport - marked" in checker


def test_mcp_release_contract_separates_legacy_tools_from_exact_modern_v3() -> None:
    mcp = " ".join((ROOT / "docs" / "mcp.md").read_text(encoding="utf-8").split())
    support = " ".join(
        (ROOT / "docs" / "support_matrix.md").read_text(encoding="utf-8").split()
    )
    release = " ".join(
        (ROOT / "docs" / "release_status.md").read_text(encoding="utf-8").split()
    )

    for required in (
        "Python MCP SDK v2",
        "protocol revisions are date strings",
        "`2026-07-28`",
        "Manifest v1 is a compatibility contract",
        "Manifest v2",
        "Manifest v3 is a new, non-downgradable authority contract",
        "`legacy`",
        "`auto`",
        "Manifest v1/v2 is the released compatibility contract for governed MCP Tools",
        "Manifest v3 is the exact `2026-07-28` modern Host-client contract",
        "governed Tools with the modern closed result union",
        "the Store uses schema v7",
        "`mcp==2.0.0`",
    ):
        assert required in mcp

    for required in (
        "Client-only Manifest v1/v2 governed Tools compatibility plus exact-`2026-07-28` Manifest v3 Tools",
        "Host-preconfigured OAuth",
        "optional digest-pinned Tasks extension",
    ):
        assert required in support

    for required in (
        "Manifest v1/v2 retains its governed Tools compatibility contract",
        "exact-`2026-07-28` Manifest v3 adds governed Tools",
        "Host-preconfigured OAuth",
        "digest-pinned Tasks extension",
    ):
        assert required in release

    for excluded in (
        "Apps",
        "Roots",
        "Sampling",
        "Logging",
    ):
        assert excluded in mcp
        assert excluded in support
        assert excluded in release
    for excluded in (
        "OAuth Dynamic Client Registration",
        "deprecated standalone HTTP+SSE transport",
        "MCP server surface",
    ):
        assert excluded in mcp
    for excluded in (
        "client-credentials",
        "enterprise-managed",
        "DPoP",
        "workload-identity",
    ):
        assert excluded in mcp
        assert excluded in support
        assert excluded in release
    assert "Ubuntu Python 3.11 and 3.14" in support


def test_mcp_extra_artifact_smoke_uses_clean_installed_modern_runtime() -> None:
    smoke = (ROOT / "scripts" / "smoke_mcp_extra.py").read_text(encoding="utf-8")

    for required in (
        'installed_module = Path(agent_libos.__file__).resolve()',
        'installed_prefix = Path(sys.prefix).resolve()',
        'installed_module.is_relative_to(installed_prefix)',
        'if installed["mcp"] != "2.0.0"',
        'server.run("stdio")',
        '"schema_version": 3',
        '"protocol_mode": "2026-07-28"',
        'runtime.mcp.register_server(',
        'runtime.mcp.list_resources(',
        'runtime.mcp.list_resource_templates(',
        'runtime.mcp.read_resource(',
        'runtime.mcp.list_prompts(',
        'runtime.mcp.get_prompt(',
        'runtime.mcp.complete_prompt(',
        'runtime.mcp.start_subscription(',
        'runtime.mcp.subscription_events(',
        'runtime.mcp.stop_subscription(',
        'runtime.mcp.call_tool(',
        'runtime.mcp.stdio_resource_for_argv(',
        '"streamable-http"',
        'plan_store_v7_migration(',
        'apply_store_v7_migration(',
        'PinnedMcpOAuthHttpTransport(',
        '_INSTALLED_OAUTH_TLS_SERVER = r\'\'\'',
        'fixture.write_text(_INSTALLED_OAUTH_TLS_SERVER',
        'runtime.mcp.add_oauth_profile(',
        'runtime.mcp.auth_begin(',
        'runtime.mcp.auth_complete(',
        '"primitive.mcp.resources.list"',
        '"primitive.mcp.resource_templates.list"',
        '"primitive.mcp.resources.read"',
        '"primitive.mcp.prompts.list"',
        '"primitive.mcp.prompts.get"',
        '"primitive.mcp.completion.complete"',
        '"primitive.mcp.subscriptions.start"',
        '"primitive.mcp.subscriptions.events"',
        '"primitive.mcp.subscriptions.stop"',
        '"primitive.mcp.call"',
        '"resource_template": "greeting/name"',
        '"runtime-v3-stdio": runtime_smoke',
        '"store-v6-to-v7": migration_smoke',
        '"oauth-tls-pkce-bearer": oauth_smoke',
        'installed MCP HTTP fixture emitted stderr',
        'installed MCP OAuth/TLS fixture emitted stderr',
    ):
        assert required in smoke
    assert "verify=False" not in smoke
    assert "pytest.skip" not in smoke
    assert 'tests" / "fixtures"' not in smoke


def test_mcp_durable_artifact_smoke_uses_installed_runtime_store_and_cli() -> None:
    smoke = (ROOT / "scripts" / "smoke_mcp_installed_mrtr_tasks.py").read_text(
        encoding="utf-8"
    )

    for required in (
        "installed_module.is_relative_to(installed_prefix)",
        "first = Runtime.open(",
        "reopened = Runtime.open(",
        "first.mcp.call_tool(",
        '"continuations",',
        '"remote-tasks",',
        "_assert_private_values_absent(root_path)",
        "if len(initial_provider.calls) != 4",
        "if continuation_provider.calls != 1",
        "tasks_provider.get_calls != 3",
        '"runtime-v3-durable-cli": _installed_smoke()',
    ):
        assert required in smoke
    assert "pytest.skip" not in smoke


def test_public_overview_records_corrected_security_and_release_boundaries() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    support = (ROOT / "docs" / "support_matrix.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())
    normalized_support = " ".join(support.split())

    for required in (
        "Trusted Runtime artifact publication is a narrow TCB exception",
        "typed Git, filesystem writes",
        "Any recognized transparent executable-launcher",
        "## Real LLM Configuration",
        "### Upgrading stores that contain Tool Groups",
        "benchmarks/builtin_tool_skills/README.md",
        "benchmarks/long_horizon_agent/README.md",
        "### Historical references (not current contracts)",
    ):
        assert required in normalized_readme
    for required in (
        "typed Git, filesystem writes",
        "static compile",
        "100k external-effect recovery gate",
        "10k runtime-publication recovery gate",
    ):
        assert required in normalized_support


def test_python_wheel_scope_is_the_core_package() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["agent_libos"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "The Python wheel contains the core `agent_libos` package" in readme


def test_release_artifacts_require_the_complete_builtin_skill_catalog() -> None:
    migration_module = "agent_libos/storage/tool_skill_migration.py"
    assert migration_module in WHEEL_REQUIRED_FILES
    assert migration_module in SDIST_REQUIRED_FILES
    assert BUILTIN_SKILL_IDS == RUNTIME_BUILTIN_SKILL_IDS
    assert (
        _BUILTIN_SKILL_MAX_INSTRUCTION_BYTES
        == RUNTIME_BUILTIN_SKILL_MAX_INSTRUCTION_BYTES
    )
    assert _BUILTIN_SKILL_MAX_FILE_BYTES == RUNTIME_BUILTIN_SKILL_MAX_FILE_BYTES
    assert _BUILTIN_SKILL_MAX_INSTRUCTION_BYTES == 16 * 1_024
    assert _BUILTIN_SKILL_MAX_FILE_BYTES == 24 * 1_024
    assert _BUILTIN_SKILL_MAX_FILE_BYTES > _BUILTIN_SKILL_MAX_INSTRUCTION_BYTES
    assert len(BUILTIN_SKILL_ARCHIVE_PATHS) == 26
    assert BUILTIN_SKILL_ARCHIVE_PATHS <= WHEEL_REQUIRED_FILES
    assert BUILTIN_SKILL_ARCHIVE_PATHS <= SDIST_REQUIRED_FILES

    payloads = {
        path: (ROOT / path).read_bytes()
        for path in BUILTIN_SKILL_ARCHIVE_PATHS
    }
    _validate_builtin_skill_archive_payloads(payloads)


def test_release_artifacts_close_over_modern_mcp_runtime_and_evidence_files() -> None:
    package_files = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "agent_libos" / "mcp").rglob("*")
        if path.is_file()
        and path.suffix == ".py"
        and "__pycache__" not in path.parts
    }
    storage_contract_files = {
        "agent_libos/storage/mcp_v7.py",
        "agent_libos/storage/mcp_v7_migration.py",
        "agent_libos/storage/postgres_schema_contract.py",
        "agent_libos/storage/postgres_schema_manifest.json",
        "agent_libos/storage/v7_schema_contract.py",
    }
    assert MCP_WHEEL_REQUIRED_FILES == package_files | storage_contract_files
    assert MCP_WHEEL_REQUIRED_FILES <= WHEEL_REQUIRED_FILES
    assert WHEEL_REQUIRED_FILES <= SDIST_REQUIRED_FILES

    expected_sdist_tree_files: set[str] = set()
    for prefix in ("examples/mcp/", "tests/fixtures/mcp_sdk_v2/"):
        expected = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / prefix).rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and "node_modules" not in path.parts
            and path.suffix in {".json", ".md", ".mjs", ".py", ".yaml"}
            and not any(part.startswith(".") for part in path.relative_to(ROOT).parts)
        }
        expected_sdist_tree_files.update(expected)
    expected_sdist_support_files = {
        "docs/mcp.md",
        "scripts/check_mcp_test_closure.py",
        "scripts/mcp_conformance_oauth_harness.mts",
        "scripts/mcp_dx.py",
        "scripts/mcp_test_support.py",
        "scripts/run_mcp_conformance.py",
        "scripts/smoke_mcp_extra.py",
        "scripts/smoke_mcp_installed_mrtr_tasks.py",
    }
    assert MCP_SDIST_REQUIRED_FILES == (
        expected_sdist_tree_files | expected_sdist_support_files
    )
    assert MCP_SDIST_REQUIRED_FILES <= SDIST_REQUIRED_FILES
    assert not any("node_modules" in path for path in SDIST_REQUIRED_FILES)


def test_release_builtin_skill_validation_rejects_missing_or_unparseable_package() -> None:
    payloads = {
        path: (ROOT / path).read_bytes()
        for path in BUILTIN_SKILL_ARCHIVE_PATHS
    }
    missing_path = min(payloads)
    without_one = dict(payloads)
    without_one.pop(missing_path)
    with pytest.raises(ValueError, match="catalog mismatch"):
        _validate_builtin_skill_archive_payloads(without_one)

    malformed = dict(payloads)
    malformed[missing_path] = b"---\nname: broken\n---\n"
    with pytest.raises(ValueError, match="missing frontmatter fields"):
        _validate_builtin_skill_archive_payloads(malformed)

    oversized = dict(payloads)
    header, _separator, _body = payloads[missing_path].partition(b"\n---\n")
    oversized[missing_path] = (
        header
        + b"\n---\n"
        + b"x" * (_BUILTIN_SKILL_MAX_INSTRUCTION_BYTES + 1)
    )
    with pytest.raises(ValueError, match="instructions exceed archive size limit"):
        _validate_builtin_skill_archive_payloads(oversized)

    oversized_file = dict(payloads)
    oversized_file[missing_path] = b"x" * (_BUILTIN_SKILL_MAX_FILE_BYTES + 1)
    with pytest.raises(ValueError, match="exceeds archive size limit"):
        _validate_builtin_skill_archive_payloads(oversized_file)


def test_readme_clean_install_smoke_covers_wheel_and_source_distribution() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "dist/agent_libos-1.5.0-py3-none-any.whl" in readme
    assert "dist/agent_libos-1.5.0.tar.gz" in readme
    assert readme.count("--require-hashes") >= 3
    assert "--no-deps dist/agent_libos-1.5.0-py3-none-any.whl" in readme
    assert "--no-deps --no-build-isolation dist/agent_libos-1.5.0.tar.gz" in readme
    for entrypoint in EXPECTED_CONSOLE_SCRIPTS:
        assert readme.count(f"/{entrypoint} --help") >= 2
    assert readme.count("uv pip check --python /tmp/agent-libos-") >= 2


def test_declared_python_support_has_an_explicit_upper_bound() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["requires-python"] == ">=3.11,<3.15"
    lock_header = (ROOT / "uv.lock").read_text(encoding="utf-8").splitlines()[:4]
    assert 'requires-python = ">=3.11, <3.15"' in lock_header


def test_gui_dependency_baseline_is_current_and_lockfile_aligned() -> None:
    package = json.loads((ROOT / "gui" / "package.json").read_text(encoding="utf-8"))
    lockfile = json.loads(
        (ROOT / "gui" / "package-lock.json").read_text(encoding="utf-8")
    )

    assert package["version"] == "1.5.0"
    assert package["dependencies"] == {
        "lucide-react": "^1.28.0",
        "react": "^19.2.8",
        "react-dom": "^19.2.8",
        "react-markdown": "^10.1.0",
        "remark-gfm": "^4.0.1",
    }
    assert package["devDependencies"] == {
        "@axe-core/playwright": "4.12.1",
        "@playwright/test": "1.62.1",
        "@testing-library/dom": "^10.4.1",
        "@testing-library/user-event": "^14.6.1",
        "@types/node": "^24.13.3",
        "@types/react": "^19.2.18",
        "@types/react-dom": "^19.2.4",
        "@vitejs/plugin-react": "^6.0.5",
        "electron": "^43.2.0",
        "jsdom": "^30.0.1",
        "typescript": "^7.0.2",
        "vite": "^8.2.0",
        "vitest": "^4.1.10",
    }
    assert package["engines"] == {
        "node": "^24.15.0 || >=26.0.0",
        "npm": ">=11",
    }
    assert "overrides" not in package
    locked_root = lockfile["packages"][""]
    assert locked_root["version"] == "1.5.0"
    assert locked_root["dependencies"] == package["dependencies"]
    assert locked_root["devDependencies"] == package["devDependencies"]
    assert locked_root["engines"] == package["engines"]
    expected_locked_versions = {
        "node_modules/@axe-core/playwright": "4.12.1",
        "node_modules/@playwright/test": "1.62.1",
        "node_modules/lucide-react": "1.28.0",
        "node_modules/react": "19.2.8",
        "node_modules/react-dom": "19.2.8",
        "node_modules/@types/node": "24.13.3",
        "node_modules/@types/react": "19.2.18",
        "node_modules/@types/react-dom": "19.2.4",
        "node_modules/@vitejs/plugin-react": "6.0.5",
        "node_modules/electron": "43.2.0",
        "node_modules/jsdom": "30.0.1",
        "node_modules/typescript": "7.0.2",
        "node_modules/vite": "8.2.0",
        "node_modules/vitest": "4.1.10",
    }
    for path, expected_version in expected_locked_versions.items():
        assert lockfile["packages"][path]["version"] == expected_version
    playwright_version = lockfile["packages"]["node_modules/@playwright/test"]["version"]
    assert lockfile["packages"]["node_modules/playwright"]["version"] == playwright_version
    assert lockfile["packages"]["node_modules/playwright-core"]["version"] == playwright_version
    assert "node_modules/esbuild" not in lockfile["packages"]
    postcss_version = lockfile["packages"]["node_modules/postcss"]["version"]
    assert tuple(map(int, postcss_version.split("."))) >= (8, 5, 23)
    development = " ".join(
        (ROOT / "docs" / "development.md").read_text(encoding="utf-8").split()
    )
    support = " ".join(
        (ROOT / "docs" / "support_matrix.md").read_text(encoding="utf-8").split()
    )
    release_status = " ".join(
        (ROOT / "docs" / "release_status.md").read_text(encoding="utf-8").split()
    )
    assert "GUI package declares Node `^24.15.0 || >=26.0.0`" in development
    assert "npm 11 or newer" in development
    assert "Per-change CI exercises the Node 24 LTS line" in development
    assert "Node `^24.15.0 || >=26.0.0` and npm `>=11`" in support
    assert "Node 26 Current satisfies the engine contract" in support
    assert "Node `^24.15.0 || >=26.0.0` and npm `>=11`" in release_status
    assert "Node 26 Current satisfies the engine contract" in release_status
    assert ">=22.12.0" not in development + support + release_status


def test_local_release_recipe_keeps_the_release_only_environment() -> None:
    development = (ROOT / "docs" / "development.md").read_text(encoding="utf-8")
    recipe = development.split("## Release Artifacts", 1)[1].split("\n## ", 1)[0]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for documented_recipe in (recipe, readme):
        assert "uv sync --frozen --no-dev --group release" in documented_recipe
        assert documented_recipe.count(
            ".venv/bin/python scripts/check_release_artifacts.py dist"
        ) == 2
        assert (
            "uv run python scripts/check_release_artifacts.py"
            not in documented_recipe
        )
    assert "uv run --frozen --no-dev --group release twine check" in recipe
    assert (
        "uv run --frozen --no-dev --group release check-wheel-contents"
        in recipe
    )
    assert "clean-install matrix is downstream of that build" in recipe


def test_gui_docs_cover_the_capability_inventory_page_contract() -> None:
    gui = " ".join((ROOT / "docs" / "gui.md").read_text(encoding="utf-8").split())

    for required in (
        "GET /api/capabilities?mode=page",
        "opaque `after` cursor",
        "{items, next_after, has_more}",
        "cannot exceed `capability.list_limit`",
        "legacy array response",
        "only one bounded page",
    ):
        assert required in gui


def test_gui_docs_cover_actor_snapshot_and_llm_counter_contracts() -> None:
    gui = " ".join((ROOT / "docs" / "gui.md").read_text(encoding="utf-8").split())

    actor_contract = gui.split("Only the checkpoint", 1)[1].split(
        "Closing the GUI server", 1
    )[0]
    for required in (
        "MCP registration, and MCP protocol-discovery endpoints",
        "omitting `actor` runs in GUI Host/admin mode",
        "supplying a non-empty process id opts into process-authority mode",
    ):
        assert required in actor_contract

    snapshot_contract = gui.split("Top-level snapshot collections", 1)[1].split(
        "## Semantic Panel", 1
    )[0]
    for required in (
        "Durable Task Runs",
        "`snapshot_collection_max_items + 1`",
        (
            "`llm_call_count` and `token_total` take the maximum of the durable "
            "hierarchical resource counters and the bounded recent-window values"
        ),
    ):
        assert required in snapshot_contract

    discovery_contract = gui.split("The MCP panel exposes", 1)[1]
    for required in (
        "Discover action omits `actor` and therefore uses GUI Host/admin authority",
        "supply a non-empty process `actor`",
        "requires the process capability for the discovery read",
        "without adding or bypassing a `confirmed: true` boundary",
    ):
        assert required in discovery_contract


def test_release_docs_distinguish_windows_ci_from_remaining_environment_gates(
) -> None:
    status = " ".join(
        (ROOT / "docs" / "release_status.md").read_text(encoding="utf-8").split()
    )
    support = " ".join(
        (ROOT / "docs" / "support_matrix.md").read_text(encoding="utf-8").split()
    )

    for required in (
        "the complete deterministic matrix in per-lane jobs on Windows 3.11",
        "checked-in CI coverage, not a separate local Windows run",
        (
            "ConPTY has no Job Object parent-death containment or wall/CPU/RSS "
            "supervisor"
        ),
        "plus Deno's `KILL_ON_JOB_CLOSE` parent-death containment",
        "Deterministic local Git path/locking tests run in Windows CI",
    ):
        assert required in status
    for required in (
        "including the native Windows `KILL_ON_JOB_CLOSE` containment path",
        "The checked-in Windows 3.11 jobs are CI evidence",
        "they are not a claim of a separate local Windows run",
        "real Git credential-manager interoperability",
    ):
        assert required in support
    assert "Native macOS and Windows process containment" not in status
    assert (
        "native Windows Git path/locking behavior are environment gates" not in status
    )


def test_console_entrypoint_registry_and_wheel_contract_are_exact() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"] == EXPECTED_CONSOLE_SCRIPTS

    valid = "\n".join(
        ["[console_scripts]"]
        + [f"{name} = {target}" for name, target in EXPECTED_CONSOLE_SCRIPTS.items()]
    ).encode()
    _validate_console_scripts(valid)

    for invalid in (
        valid.replace(b"agent-libos-migrate-tool-groups = ", b"removed = "),
        valid.replace(b"agent_libos.api.cli:cli", b"agent_libos.api.cli:main"),
        valid + b"\nunexpected = package.module:main\n",
    ):
        with pytest.raises(ValueError, match="wheel console entry points mismatch"):
            _validate_console_scripts(invalid)


def test_release_documentation_covers_every_console_script() -> None:
    for path in (
        ROOT / "README.md",
        ROOT / "docs" / "cli.md",
        ROOT / "docs" / "development.md",
        ROOT / "docs" / "release_status.md",
    ):
        text = path.read_text(encoding="utf-8")
        for entrypoint in EXPECTED_CONSOLE_SCRIPTS:
            assert entrypoint in text, f"{path.relative_to(ROOT)} omits {entrypoint}"


def test_anonymity_checklist_covers_exact_tree_history_and_binary_artifacts() -> None:
    text = (ROOT / "docs" / "artifact_anonymity.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    for required in (
        "set -Eeuo pipefail",
        "set -o pipefail",
        "git status --porcelain=v1 --untracked-files=all",
        "git submodule status --recursive",
        "git for-each-ref",
        'ANON_SCAN_DIR="$(mktemp -d "$ANON_REVIEW_DIR/scan.XXXXXX")"',
        "capture_matches()",
        '0) test -s "$destination"',
        '1) test ! -s "$destination"',
        'raw = Path(sys.argv[1]).read_bytes().split(b"\\0")',
        'if value != b"lfs"',
        "unsupported Git content filter",
        "Git LFS paths are unsupported by this runbook",
        "flatten reviewed LFS bytes, remove the filter, commit, and restart",
        "git --no-replace-objects rev-list --objects --all",
        "git --no-replace-objects cat-file",
        "git --no-replace-objects cat-file --batch-all-objects",
        'case "$object_type" in',
        'commit|tag|tree|blob)',
        "rg -a -qi",
        'ANON_COMMIT_TREE="$ANON_SCAN_DIR/exact-commit-tree"',
        'ANON_COMMIT_EXPORT="$ANON_SCAN_DIR/archive-projection"',
        'git --no-replace-objects archive --format=tar "$ANON_COMMIT"',
        "scan_root commit-tree",
        "scan_root archive-projection",
        "scan_root generated-output",
        "candidate inventory changed; discard the old review and restart",
        "review is required; candidates persisted at",
        '"second_reviewer"',
        'row.get("disposition") != "intentional non-identifying fixture"',
        "non-regular archive member",
        'archive.extractfile(member)',
        'python3 "$ANON_SCAN_DIR/build_final_tar.py"',
        'python3 "$ANON_SCAN_DIR/safe_tar.py"',
        "source.extracted.inventory.json",
        "generated.extracted.inventory.json",
        "final-extract.mime-inventory",
        "find \"$ANON_BINARY_ROOT\" -type f -exec file --mime-type",
        "The initial archive hash is recorded only after",
    ):
        assert required in text
    assert text.count("scan_root archive-projection") >= 2
    assert text.count("scan_root generated-output") >= 2
    assert "--batch-check='%(objectname) %(objecttype) %(objectsize) %(rest)'" in text
    assert "Structural validation is not an anonymity scan" in normalized
    assert (
        "raw commit tree, archive projection, generated output, and safely "
        "re-extracted final archive"
    ) in normalized
    assert "applies committed `export-ignore` and `export-subst` attributes" in normalized
    assert '["git", "--no-replace-objects", "cat-file", "blob", oid]' in text


def test_anonymity_runbook_closes_filter_review_path_and_inventory_gates() -> None:
    text = (ROOT / "docs" / "artifact_anonymity.md").read_text(encoding="utf-8")

    assert "rg -a -q 'filter: lfs'" not in text
    for required in (
        'read_bytes().split(b"\\0")',
        "if len(raw) % 3:",
        'if value != b"lfs":',
        'if test -f "$ANON_SCAN_DIR/lfs-required.json"; then',
        "exit 4",
        "if within(path, worktree) or within(worktree, path):",
        "if within(output, review) or within(review, output):",
        "if within(archive, output) or within(archive, review):",
        "if path.parent != review:",
        "recorded_argument = Path(sys.argv[6])",
        "if recorded_argument.is_symlink():",
        "recorded_candidates = recorded_argument.resolve(strict=False)",
        '("recorded candidates", recorded_candidates)',
        'ANON_RECORDED_CANDIDATES="$ANON_REVIEW_DIR/candidates-$ANON_COMMIT.json"',
        "if recorded.is_symlink():",
        'getattr(os, "O_NOFOLLOW", 0)',
        'descriptor = os.open(recorded, flags, 0o600)',
        'if ! test -f "$ANON_DISPOSITIONS"; then',
        "exit 3",
        '"mode": "0755" if path.is_dir() or path.stat().st_mode & 0o111 else "0644"',
        "if expected != refreshed:",
        "if expected != extracted:",
        'sha256_file() {',
        'ANON_VERIFIED_FINAL_SHA256="$(sha256_file "$ANON_FINAL_ARCHIVE")"',
        'test "$ANON_VERIFIED_FINAL_SHA256" = "$ANON_FINAL_SHA256"',
    ):
        assert required in text

    assert text.index("ANON_RECORDED_CANDIDATES=") < text.index("ANON_SCAN_DIR=")
    assert text.index("descriptor = os.open(recorded, flags, 0o600)") < text.index(
        'if ! test -f "$ANON_DISPOSITIONS"; then'
    )


def test_anonymity_prologue_rejects_dangling_candidate_symlink(
    tmp_path: Path,
) -> None:
    text = (ROOT / "docs" / "artifact_anonymity.md").read_text(encoding="utf-8")
    marker = '"$ANON_RECORDED_CANDIDATES" <<\'PY\'\n'
    guard = text.split(marker, 1)[1].split("\nPY\n", 1)[0]

    worktree = tmp_path / "worktree"
    output = tmp_path / "generated"
    review = tmp_path / "review"
    for directory in (worktree, output, review):
        directory.mkdir()
    escaped = tmp_path / "escaped-candidates.json"
    candidates = review / "candidates.json"
    try:
        candidates.symlink_to(escaped)
    except OSError as exc:
        pytest.skip(f"symbolic links unavailable: {exc}")

    completed = subprocess.run(
        [
            sys.executable,
            "-",
            str(worktree),
            str(output),
            str(review),
            str(tmp_path / "final.tar"),
            str(review / "dispositions.json"),
            str(candidates),
        ],
        input=guard,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "recorded candidate path must not be a symbolic link" in completed.stderr
    assert not escaped.exists()


def test_maintainer_governance_docs_match_current_release_boundaries() -> None:
    security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    security_normalized = " ".join(security.split())
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    releasing = (ROOT / "docs" / "releasing.md").read_text(encoding="utf-8")
    releasing_normalized = " ".join(releasing.split())
    version = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]

    for link in ("AGENTS.md", "docs/development.md", "SECURITY.md"):
        assert link in contributing
    assert "only the non-sensitive coordination request" in contributing

    assert (
        "private vulnerability reporting is currently **not enabled**"
        in security_normalized
    )
    assert "is not presently a working intake channel" in security_normalized
    assert "only a non-sensitive request" in security_normalized
    assert "does not invent an email address" in security_normalized
    assert "No current validation or fix commitment" in security_normalized
    assert "mailto:" not in security
    assert re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", security) is None

    assert "## Unreleased" in changelog
    assert f"## {version} — release candidate" in changelog
    assert "does not claim that a tag, PyPI upload, GitHub release" in changelog

    for required in (
        "read-only repository permission and no PyPI, tag, or GitHub-release authority",
        "Public publication is blocked",
        "enable GitHub private vulnerability reporting",
        "Run the command blocks in order in the same trusted Bash session",
        "Choose an unreused final-form numeric `X.Y.Z` version",
        "PEP 440 spellings that a build backend can normalize are not",
        "LOCAL_RELEASE_WHEEL=",
        "RELEASE_DOWNLOAD_DIR=/absolute/path/to/download",
        "requested.resolve(strict=True)",
        "actual != expected or len(entries) != len(expected)",
        "entry.is_symlink() or not entry.is_file()",
        'RELEASE_WHEEL="$RELEASE_DOWNLOAD_DIR/agent_libos-${RELEASE_VERSION}-py3-none-any.whl"',
        'RELEASE_SDIST="$RELEASE_DOWNLOAD_DIR/agent_libos-${RELEASE_VERSION}.tar.gz"',
        "--repository testpypi",
        "--repository pypi",
        "explicitly authorizes or rejects that exact preview",
        "yank the exact version",
        "Never overwrite or reuse a published version",
        ': "${RELEASE_COMMIT:?set the exact authorized commit}"',
        'TAG_COMMIT="$(git rev-list -n 1 "v${RELEASE_VERSION}")"',
        'test "$TAG_COMMIT" = "$RELEASE_COMMIT"',
    ):
        assert required in releasing_normalized
    bash_blocks = re.findall(r"```bash\n(.*?)\n```", releasing, flags=re.DOTALL)
    assert bash_blocks
    assert all(
        block.startswith("set -Eeuo pipefail\nIFS=$'\\n\\t'\n")
        for block in bash_blocks
    )
    assert releasing.count('"$RELEASE_WHEEL" "$RELEASE_SDIST"') == 2


def test_ci_actions_and_downloaded_toolchains_are_immutable() -> None:
    seen_actions: set[str] = set()
    for workflow_path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        raw = workflow_path.read_text(encoding="utf-8")
        for line in raw.splitlines():
            if "uses:" not in line:
                continue
            match = re.fullmatch(
                r"\s*uses:\s+([^@\s]+)@([0-9a-f]{40})\s+#\s+(v[0-9.]+)\s*",
                line,
            )
            assert match is not None, f"unversioned action in {workflow_path}: {line}"
            action, digest, version = match.groups()
            assert ACTION_PINS[action] == (digest, version)
            seen_actions.add(action)

        parsed = yaml.safe_load(raw)
        for job in parsed["jobs"].values():
            assert isinstance(job.get("timeout-minutes"), int)
            for step in job.get("steps", []):
                action = str(step.get("uses") or "").split("@", 1)[0]
                if action == "astral-sh/setup-uv":
                    assert step.get("with", {}).get("version") == "0.11.32"
                elif action == "denoland/setup-deno":
                    assert step.get("with", {}).get("deno-version") == "2.9.4"
                elif action == "actions/upload-artifact":
                    assert step.get("with", {}).get("retention-days") == 14

    assert seen_actions == set(ACTION_PINS)


def test_ci_checkout_does_not_persist_credentials_in_git_config() -> None:
    checkout_jobs: list[str] = []
    for workflow_path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        parsed = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        assert parsed["permissions"] == {"contents": "read"}
        for job_name, job in parsed["jobs"].items():
            checkout_steps = [
                step
                for step in job.get("steps", [])
                if str(step.get("uses") or "").startswith("actions/checkout@")
            ]
            if not checkout_steps:
                continue
            checkout_jobs.append(f"{workflow_path.name}:{job_name}")
            assert len(checkout_steps) == 1
            assert (
                checkout_steps[0].get("with", {}).get("persist-credentials")
                is False
            )
    assert checkout_jobs


def test_release_ci_is_a_compatibility_gate_without_publish_authority() -> None:
    workflow_text = "\n".join(
        workflow.read_text(encoding="utf-8")
        for workflow in sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    ).lower()
    for forbidden in (
        "twine upload",
        "pypa/gh-action-pypi-publish",
        "gh release create",
        "pypi-token",
    ):
        assert forbidden not in workflow_text

    support = " ".join(
        (ROOT / "docs" / "support_matrix.md").read_text(encoding="utf-8").split()
    )
    invariants = " ".join(
        (ROOT / "docs" / "invariants.md").read_text(encoding="utf-8").split()
    )
    for text in (support, invariants):
        assert "not a bit-for-bit reproducible publication chain" in text
        assert "pinned to reviewed full commit SHAs" in text
        assert "one canonical wheel/source pair" in text
    assert "performs no PyPI upload or external release mutation" in support
    assert "external release mutation remains separately authorized" in invariants


def test_release_workflow_runs_release_smokes_without_repeating_lane_matrix() -> None:
    workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text(
        encoding="utf-8"
    )
    parsed = yaml.safe_load(workflow)
    steps = parsed["jobs"]["deterministic-release"]["steps"]
    runtime_safety = next(
        step
        for step in steps
        if step.get("name") == "Run runtime-safety release smoke"
    )
    assert "if" not in runtime_safety
    assert "continue-on-error" not in runtime_safety
    release_commands = "\n".join(str(step.get("run") or "") for step in steps)
    runtime_safety_step = str(runtime_safety["run"])
    assert "scripts/test_matrix.py" not in release_commands
    assert "--lane all" not in release_commands
    assert "experiments/run_benchmark.py" in runtime_safety_step
    assert "--suite benchmarks/runtime_safety" in runtime_safety_step
    assert "--require-all-passed" in runtime_safety_step
    assert "--limit" not in runtime_safety_step


def test_benchmark_only_workflow_uses_the_frozen_runtime_environment() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "external-effect-recovery-scale.yml"
    ).read_text(encoding="utf-8")
    parsed = yaml.safe_load(workflow)
    steps = parsed["jobs"]["million-record-recovery"]["steps"]

    install_step = next(
        step for step in steps if step.get("name") == "Install dependencies"
    )
    assert install_step["run"] == "uv sync --frozen --no-dev"
    benchmark_steps = [
        step for step in steps if str(step.get("name") or "").startswith("Run ")
    ]
    assert len(benchmark_steps) == 2
    for step in benchmark_steps:
        assert str(step["run"]).startswith("uv run --frozen --no-dev python ")
    assert "--all-groups" not in workflow


def test_agentdojo_ci_uses_its_isolated_frozen_environment() -> None:
    workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text(
        encoding="utf-8"
    )
    parsed = yaml.safe_load(workflow)
    job = parsed["jobs"]["agentdojo"]

    assert job["needs"] == "static"
    assert job["strategy"]["matrix"]["python-version"] == ["3.11", "3.12"]
    lock_step = next(
        item
        for item in job["steps"]
        if item.get("name") == "Check isolated AgentDojo lockfile is current"
    )
    assert lock_step["working-directory"] == "experiments/agentdojo"
    assert lock_step["run"] == "uv lock --check"
    for step_name, command in (
        ("Install isolated AgentDojo dependencies", "uv sync --frozen"),
        ("Run isolated AgentDojo tests", "uv run --frozen pytest -q"),
    ):
        step = next(item for item in job["steps"] if item.get("name") == step_name)
        assert step["working-directory"] == "experiments/agentdojo"
        assert step["run"] == command
        assert "if" not in step
        assert "continue-on-error" not in step
    assert job["timeout-minutes"] == 15


def test_fail_on_skip_gate_changes_a_successful_session_to_failure() -> None:
    reporter = type("Reporter", (), {"stats": {"skipped": [object()]}})()

    class Config:
        rootpath = ROOT
        pluginmanager = type(
            "PluginManager",
            (),
            {"get_plugin": staticmethod(lambda name: reporter if name == "terminalreporter" else None)},
        )()

        @staticmethod
        def getoption(name: str, default: object = None) -> object:
            return {
                "--keep-agent-outputs": True,
                "--fail-on-skip": True,
            }.get(name, default)

    session = type("Session", (), {"config": Config(), "exitstatus": pytest.ExitCode.OK})()

    pytest_sessionfinish(session, int(pytest.ExitCode.OK))

    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED


def test_release_workflow_preserves_and_clean_installs_validated_artifacts() -> None:
    workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text(
        encoding="utf-8"
    )
    parsed = yaml.safe_load(workflow)
    release_job = parsed["jobs"]["release-artifacts"]
    smoke_job = parsed["jobs"]["release-artifact-smoke"]
    required_jobs = {
        "static",
        "agentdojo",
        "python",
        "security",
        "host-filesystem-identity",
        "mcp-sdk",
        "mcp-native",
        "deterministic-release",
        "postgres",
        "gui",
        "windows",
    }
    assert set(release_job["needs"]) == required_jobs
    assert smoke_job["needs"] == "release-artifacts"
    assert "if" not in release_job
    assert "continue-on-error" not in release_job
    for job_name in required_jobs:
        assert "if" not in parsed["jobs"][job_name]
        assert "continue-on-error" not in parsed["jobs"][job_name]
    python_matrix = parsed["jobs"]["python"]["strategy"]["matrix"]
    assert python_matrix["python-version"] == ["3.11", "3.14"]
    assert python_matrix["lane"] == [
        "unit",
        "self-evolution",
        "providers",
        "benchmark",
    ]
    assert python_matrix["include"] == [
        {
            "python-version": "3.11",
            "lane": "runtime",
            "shard_args": "--shard-count 2 --shard-index 0",
        },
        {
            "python-version": "3.11",
            "lane": "runtime",
            "shard_args": "--shard-count 2 --shard-index 1",
        },
        {
            "python-version": "3.14",
            "lane": "runtime",
            "shard_args": "--shard-count 2 --shard-index 0",
        },
        {
            "python-version": "3.14",
            "lane": "runtime",
            "shard_args": "--shard-count 2 --shard-index 1",
        },
    ]
    security_matrix = parsed["jobs"]["security"]["strategy"]["matrix"]
    assert security_matrix["python-version"] == ["3.11", "3.14"]
    assert "strategy" not in release_job
    assert smoke_job["strategy"]["matrix"]["python-version"] == [
        "3.11",
        "3.12",
        "3.13",
        "3.14",
    ]
    windows_job = parsed["jobs"]["windows"]
    assert windows_job["runs-on"] == "windows-latest"
    assert windows_job["needs"] == "static"
    assert windows_job["timeout-minutes"] == 30
    assert windows_job["name"] == "windows (${{ matrix.name }})"
    assert windows_job["strategy"] == {
        "fail-fast": False,
        "matrix": {
            "include": [
                {"name": "unit", "lane": "unit", "shard_args": ""},
                {
                    "name": "runtime 1/4",
                    "lane": "runtime",
                    "shard_args": "--shard-count 4 --shard-index 0",
                },
                {
                    "name": "runtime 2/4",
                    "lane": "runtime",
                    "shard_args": "--shard-count 4 --shard-index 1",
                },
                {
                    "name": "runtime 3/4",
                    "lane": "runtime",
                    "shard_args": "--shard-count 4 --shard-index 2",
                },
                {
                    "name": "runtime 4/4",
                    "lane": "runtime",
                    "shard_args": "--shard-count 4 --shard-index 3",
                },
                {"name": "security", "lane": "security", "shard_args": ""},
                {
                    "name": "self-evolution",
                    "lane": "self-evolution",
                    "shard_args": "",
                },
                {
                    "name": "providers 1/3",
                    "lane": "providers",
                    "shard_args": "--shard-count 3 --shard-index 0",
                },
                {
                    "name": "providers 2/3",
                    "lane": "providers",
                    "shard_args": "--shard-count 3 --shard-index 1",
                },
                {
                    "name": "providers 3/3",
                    "lane": "providers",
                    "shard_args": "--shard-count 3 --shard-index 2",
                },
                {
                    "name": "benchmark 1/2",
                    "lane": "benchmark",
                    "shard_args": "--shard-count 2 --shard-index 0",
                },
                {
                    "name": "benchmark 2/2",
                    "lane": "benchmark",
                    "shard_args": "--shard-count 2 --shard-index 1",
                },
            ]
        },
    }
    windows_python = next(
        item
        for item in windows_job["steps"]
        if item.get("name") == "Set up Python"
    )
    assert windows_python["with"]["python-version"] == "3.11"
    windows_install = next(
        item
        for item in windows_job["steps"]
        if item.get("name") == "Install dependencies with native PTY support"
    )
    assert windows_install["run"] == "uv sync --frozen --extra pty"
    windows_tests = next(
        item
        for item in windows_job["steps"]
        if item.get("name") == "Run deterministic Python lane"
    )
    assert windows_tests["env"]["PYTEST_ADDOPTS"] == "--timeout=300"
    host_identity_job = parsed["jobs"]["host-filesystem-identity"]
    assert host_identity_job["runs-on"] == "${{ matrix.runner }}"
    assert host_identity_job["needs"] == "static"
    assert host_identity_job["timeout-minutes"] == 15
    assert host_identity_job["strategy"] == {
        "fail-fast": False,
        "matrix": {
            "include": [
                {
                    "runner": "ubuntu-latest",
                    "marker": "platform_linux",
                },
                {
                    "runner": "macos-14",
                    "marker": "platform_darwin",
                },
            ]
        },
    }
    host_identity_python = next(
        item
        for item in host_identity_job["steps"]
        if item.get("name") == "Set up Python"
    )
    assert host_identity_python["with"]["python-version"] == "3.11"
    host_identity_install = next(
        item
        for item in host_identity_job["steps"]
        if item.get("name") == "Install dependencies"
    )
    assert host_identity_install["run"] == "uv sync --frozen"
    host_identity_test = next(
        item
        for item in host_identity_job["steps"]
        if item.get("name") == "Run native host filesystem identity gate"
    )
    host_identity_command = str(host_identity_test["run"])
    for required in (
        "tests/security/test_filesystem_path_identity.py",
        "-m ${{ matrix.marker }}",
        "--fail-on-skip",
    ):
        assert required in host_identity_command
    assert "-k " not in host_identity_command
    assert "--ignore" not in host_identity_command
    root_lock_step = next(
        item
        for item in parsed["jobs"]["static"]["steps"]
        if item.get("name") == "Check root lockfile is current"
    )
    assert root_lock_step["run"] == "uv lock --check"
    static_steps = parsed["jobs"]["static"]["steps"]
    static_install = next(
        item for item in static_steps if item.get("name") == "Install dependencies"
    )
    assert static_install["run"] == "uv sync --frozen"
    assert all(item.get("name") != "Set up Deno" for item in static_steps)
    assert all(
        item.get("name") != "Set up Deno"
        for item in parsed["jobs"]["deterministic-release"]["steps"]
    )
    expected_dependency_installs = {
        "static": ("Install dependencies", "uv sync --frozen"),
        "python": ("Install dependencies", "uv sync --frozen"),
        "security": ("Install dependencies", "uv sync --frozen"),
        "host-filesystem-identity": (
            "Install dependencies",
            "uv sync --frozen",
        ),
        "mcp-sdk": (
            "Install frozen MCP SDK dependencies",
            "uv sync --frozen --extra mcp",
        ),
        "mcp-native": (
            "Install frozen MCP SDK dependencies",
            "uv sync --frozen --extra mcp",
        ),
        "windows": (
            "Install dependencies with native PTY support",
            "uv sync --frozen --extra pty",
        ),
        "deterministic-release": (
            "Install dependencies",
            "uv sync --frozen --no-dev",
        ),
        "postgres": (
            "Install dependencies",
            "uv sync --frozen --extra postgres",
        ),
    }
    for job_name, (step_name, expected_command) in expected_dependency_installs.items():
        install_step = next(
            item
            for item in parsed["jobs"][job_name]["steps"]
            if item.get("name") == step_name
        )
        assert install_step["run"] == expected_command
    assert "--all-groups" not in workflow
    for job_name in (
        "python",
        "security",
        "windows",
        "mcp-sdk",
    ):
        deno_step = next(
            item
            for item in parsed["jobs"][job_name]["steps"]
            if item.get("name") == "Set up Deno"
        )
        expected_sha, _ = ACTION_PINS["denoland/setup-deno"]
        assert deno_step["uses"] == f"denoland/setup-deno@{expected_sha}"
        assert deno_step["with"]["deno-version"] == "2.9.4"
        assert "if" not in deno_step
        assert "continue-on-error" not in deno_step
    critical_upstream_steps = (
        ("static", "Compile Python sources", "python -m compileall"),
        ("static", "Check architecture and blocking-work boundaries", "scripts/check_architecture.py"),
        ("static", "Check protected-operation coverage", "scripts/check_protected_operations.py"),
        ("static", "Check invariant manifest and pytest collection", "scripts/check_test_invariants.py"),
        ("python", "Run pytest lane", "scripts/test_matrix.py --lane"),
        ("security", "Run complete security lane", "scripts/test_matrix.py --lane security"),
        (
            "host-filesystem-identity",
            "Run native host filesystem identity gate",
            (
                "tests/security/test_filesystem_path_identity.py",
                "-m ${{ matrix.marker }}",
                "--fail-on-skip",
            ),
        ),
        (
            "deterministic-release",
            "Run runtime-safety release smoke",
            "--require-all-passed",
        ),
        (
            "deterministic-release",
            "Run 100k external-effect recovery scale smoke",
            "experiments/run_external_effect_recovery_scale.py",
        ),
        (
            "deterministic-release",
            "Run 10k runtime-publication handler scale smoke",
            "experiments/run_publication_reconciliation_scale.py",
        ),
        (
            "postgres",
            "Run PostgreSQL store integration tests",
            "-m postgres --run-postgres",
        ),
        (
            "mcp-sdk",
            "Run complete MCP SDK integration suite",
            (
                "tests -m \"mcp and not postgres\"",
                "--run-mcp --fail-on-skip",
            ),
        ),
        (
            "mcp-sdk",
            "Run reviewed fixed-upstream MCP client conformance scenarios",
            "scripts/run_mcp_conformance.py",
        ),
        (
            "mcp-native",
            "Run native stdio and loopback HTTP MCP smoke",
            (
                "test_stdio_fastmcp_tool_call",
                "test_stdio_modern_discovery_call_and_phase_receipts",
                "test_streamable_http_fastmcp_tool_call",
                "--run-mcp --fail-on-skip",
            ),
        ),
        (
            "gui",
            "Install GUI E2E runtime",
            (
                "uv sync --frozen",
                "npm --prefix gui exec -- playwright install --with-deps chromium",
            ),
        ),
        (
            "gui",
            "Run GUI checks",
            (
                "npm --prefix gui run test",
                "npm --prefix gui run typecheck",
                "npm --prefix gui run build",
                "npm --prefix gui run test:e2e",
            ),
        ),
        (
            "windows",
            "Run deterministic Python lane",
            (
                "scripts/test_matrix.py --lane ${{ matrix.lane }}",
                "${{ matrix.shard_args }}",
                "--max-lane-seconds 1400",
            ),
        ),
    )
    for job_name, step_name, command_fragments in critical_upstream_steps:
        step = next(
            item
            for item in parsed["jobs"][job_name]["steps"]
            if item.get("name") == step_name
        )
        assert "if" not in step
        assert "continue-on-error" not in step
        expected_fragments = (
            (command_fragments,)
            if isinstance(command_fragments, str)
            else command_fragments
        )
        assert all(fragment in str(step["run"]) for fragment in expected_fragments)
        if job_name in {"python", "security", "deterministic-release"}:
            assert "--skip-real-deno" not in str(step["run"])
    for job_name, step_name, lane_deadline in (
        (
            "python",
            "Run pytest lane",
            "${{ matrix.lane == 'runtime' && 480 || 360 }}",
        ),
        ("security", "Run complete security lane", "360"),
    ):
        step = next(
            item
            for item in parsed["jobs"][job_name]["steps"]
            if item.get("name") == step_name
        )
        assert step["timeout-minutes"] == 15
        command = str(step["run"])
        assert "--durations 25" in command
        assert f"--max-lane-seconds {lane_deadline}" in command
    postgres_job = parsed["jobs"]["postgres"]
    postgres_service = postgres_job["services"]["postgres"]
    assert postgres_service["image"] == (
        "postgres:17.10-bookworm@sha256:"
        "4f736ae292687621d4dbe0d499ffd024a36bd2ee7d8ca6f2ccd4c800f047b394"
    )
    assert postgres_service["ports"] == ["5432:5432"]
    postgres_step = next(
        item
        for item in postgres_job["steps"]
        if item.get("name") == "Run PostgreSQL store integration tests"
    )
    assert postgres_step["env"]["AGENT_LIBOS_POSTGRES_DSN"] == (
        "postgresql://agent_libos:agent_libos@127.0.0.1:5432/agent_libos"
    )
    assert "--fail-on-skip" in str(postgres_step["run"])
    mcp_job = parsed["jobs"]["mcp-sdk"]
    assert mcp_job["strategy"] == {
        "fail-fast": False,
        "matrix": {"python-version": ["3.11", "3.14"]},
    }
    assert mcp_job["name"] == "mcp sdk (${{ matrix.python-version }})"
    mcp_python = next(
        item
        for item in mcp_job["steps"]
        if item.get("name") == "Set up Python"
    )
    assert mcp_python["with"]["python-version"] == "${{ matrix.python-version }}"
    mcp_node = next(
        item
        for item in mcp_job["steps"]
        if item.get("name") == "Set up Node"
    )
    assert mcp_node["with"]["node-version"] == "24"
    mcp_install = next(
        item
        for item in mcp_job["steps"]
        if item.get("name") == "Install frozen MCP SDK dependencies"
    )
    assert mcp_install["run"] == "uv sync --frozen --extra mcp"
    mcp_typescript_install = next(
        item
        for item in mcp_job["steps"]
        if item.get("name") == "Install frozen TypeScript MCP SDK v2 fixture"
    )
    assert (
        "npm --prefix tests/fixtures/mcp_sdk_v2/typescript_server ci "
        "--ignore-scripts --no-audit --no-fund"
    ) == mcp_typescript_install["run"]
    assert "if" not in mcp_typescript_install
    assert "continue-on-error" not in mcp_typescript_install
    mcp_test = next(
        item
        for item in mcp_job["steps"]
        if item.get("name") == "Run complete MCP SDK integration suite"
    )
    mcp_command = str(mcp_test["run"])
    assert 'tests -m "mcp and not postgres"' in mcp_command
    assert "--run-mcp --fail-on-skip" in mcp_command
    assert "-k " not in mcp_command
    assert "--ignore" not in mcp_command
    mcp_closure = next(
        item
        for item in mcp_job["steps"]
        if item.get("name") == "Verify complete MCP pytest marker closure"
    )
    assert mcp_closure["run"] == (
        "uv run python scripts/check_mcp_test_closure.py"
    )
    assert "if" not in mcp_closure
    assert "continue-on-error" not in mcp_closure
    mcp_conformance = next(
        item
        for item in mcp_job["steps"]
        if item.get("name")
        == "Run reviewed fixed-upstream MCP client conformance scenarios"
    )
    assert mcp_conformance["run"] == "uv run python scripts/run_mcp_conformance.py"
    assert "expected-failures" not in str(mcp_conformance["run"])
    assert "if" not in mcp_conformance
    assert "continue-on-error" not in mcp_conformance
    mcp_native_job = parsed["jobs"]["mcp-native"]
    assert mcp_native_job["needs"] == "static"
    assert mcp_native_job["runs-on"] == "${{ matrix.os }}"
    assert mcp_native_job["strategy"] == {
        "fail-fast": False,
        "matrix": {"os": ["windows-latest", "macos-14"]},
    }
    assert mcp_native_job["name"] == "mcp native transport (${{ matrix.os }})"
    native_test = next(
        item
        for item in mcp_native_job["steps"]
        if item.get("name") == "Run native stdio and loopback HTTP MCP smoke"
    )
    native_command = str(native_test["run"])
    assert native_command.count("test_mcp_sdk_integration.py::") == 3
    for modern_cli_test in (
        "test_modern_cli_runs_real_stdio_and_loopback_http_without_stderr",
        "test_oauth_foreground_login_rehydrates_in_fresh_cli_runtime_and_resources",
        "test_subscription_listen_streams_and_stops_on_one_real_runtime",
    ):
        assert (
            "tests/providers/test_mcp_modern_cli_e2e.py::" + modern_cli_test
            in native_command
        )
    for durable_cli_test in (
        "test_runtime_facade_captures_real_human_and_reopens_without_initial_replay",
        "test_cli_resource_and_prompt_initial_elicitation_use_protected_host_questions",
        "test_runtime_remote_task_facade_get_update_cancel_uses_only_local_refs",
    ):
        assert (
            "tests/runtime/test_mcp_v3_durable_facade.py::" + durable_cli_test
            in native_command
        )
    assert "--run-mcp --fail-on-skip" in native_command
    assert "skip" not in native_command.removesuffix("--fail-on-skip")
    assert "if" not in native_test
    assert "continue-on-error" not in native_test
    compile_step = next(
        item
        for item in parsed["jobs"]["static"]["steps"]
        if item.get("name") == "Compile Python sources"
    )
    assert "compileall agent_libos tests scripts experiments benchmarks modules" in str(
        compile_step["run"]
    )
    assert parsed["concurrency"] == {
        "group": "tests-${{ github.event.pull_request.number || github.ref }}",
        "cancel-in-progress": True,
    }
    assert parsed["env"] == {
        "RELEASE_WHEEL": "dist/agent_libos-1.5.0-py3-none-any.whl",
        "RELEASE_SDIST": "dist/agent_libos-1.5.0.tar.gz",
        "RELEASE_CHECKSUMS": "dist/SHA256SUMS",
    }
    release_steps = release_job["steps"]
    build_step = next(
        step
        for step in release_steps
        if step.get("name") == "Build and validate distributions"
    )
    assert "if" not in build_step
    assert "continue-on-error" not in build_step
    build_command = str(build_step["run"])
    assert workflow.count(
        "uv build --no-build-isolation --clear --out-dir dist"
    ) == 1
    assert "--python .venv/bin/python" in build_command
    assert "--no-create-gitignore" in build_command
    release_sync = next(
        step
        for step in release_steps
        if step.get("name") == "Install frozen release build tools"
    )
    assert release_sync["run"] == "uv sync --frozen --no-dev --group release"
    assert "python scripts/check_release_artifacts.py dist --write-checksums" in build_command
    assert (
        'uv run --frozen --no-dev --group release twine check "$RELEASE_WHEEL" '
        '"$RELEASE_SDIST"'
    ) in build_command
    assert (
        'uv run --frozen --no-dev --group release check-wheel-contents '
        '"$RELEASE_WHEEL"'
    ) in build_command
    assert "python scripts/check_release_artifacts.py dist --verify-checksums" in build_command
    assert "dist/*" not in workflow
    assert "find dist" not in workflow
    runtime_safety_step = next(
        str(step["run"])
        for step in parsed["jobs"]["deterministic-release"]["steps"]
        if step.get("name") == "Run runtime-safety release smoke"
    )
    assert "experiments/run_benchmark.py" in runtime_safety_step
    assert "--require-all-passed" in runtime_safety_step
    assert "--limit" not in runtime_safety_step
    benchmark_only_steps = {
        step["name"]: str(step["run"])
        for step in parsed["jobs"]["deterministic-release"]["steps"]
        if str(step.get("name") or "").startswith("Run ")
    }
    assert set(benchmark_only_steps) == {
        "Run runtime-safety release smoke",
        "Run 100k external-effect recovery scale smoke",
        "Run 10k runtime-publication handler scale smoke",
        "Run Durable TaskRun six-barrier crash gate",
        "Run 100k Durable TaskRun recovery scale gate",
    }
    for command in benchmark_only_steps.values():
        assert command.startswith("uv run --frozen --no-dev python ")
    smoke_steps = smoke_job["steps"]
    download_step = next(
        step
        for step in smoke_steps
        if step.get("name") == "Download canonical candidate distributions"
    )
    expected_download_sha, _ = ACTION_PINS["actions/download-artifact"]
    assert download_step["uses"] == (
        f"actions/download-artifact@{expected_download_sha}"
    )
    assert download_step["with"] == {
        "name": "agent-libos-canonical-${{ github.sha }}-${{ github.run_attempt }}",
        "path": "dist",
    }
    verify_step = next(
        step
        for step in smoke_steps
        if step.get("name") == "Verify canonical candidate distributions"
    )
    assert "--verify-checksums" in str(verify_step["run"])
    assert "sha256sum --check --strict SHA256SUMS" in str(verify_step["run"])
    export_step = next(
        step
        for step in smoke_steps
        if step.get("name") == "Export frozen artifact dependency sets"
    )
    export_command = str(export_step["run"])
    assert (
        "uv export --frozen --no-dev --extra mcp --no-emit-project "
        "--output-file mcp-runtime-requirements.txt"
    ) in export_command
    assert "uv export --frozen --only-group release --no-emit-project" in export_command
    wheel_install_step = next(
        step
        for step in smoke_steps
        if step.get("name") == "Clean-install wheel and run entrypoint smoke"
    )
    sdist_install_step = next(
        step
        for step in smoke_steps
        if step.get("name") == "Clean-install source distribution"
    )
    for step in (wheel_install_step, sdist_install_step):
        assert "if" not in step
        assert "continue-on-error" not in step
        command = str(step["run"])
        assert 'uv venv --python "${{ matrix.python-version }}"' in command
        assert "find dist" not in command
        assert "--require-hashes" in command
        assert "--no-deps" in command
        assert "mcp-runtime-requirements.txt" in command
        assert "scripts/smoke_mcp_extra.py" in command
        assert "scripts/smoke_mcp_installed_mrtr_tasks.py" in command
        assert "[mcp]" in command
        assert "get_builtin_skill_catalog" in command
        assert "len(catalog.list()) == 26" in command
        assert "len(owned) == 101" in command
        assert "runtime.tools.list()" in command
        assert "owned == registered" in command
        assert "runtime.skills.discover_skills_result" in command
        assert "'expected_package_sha256': selected['package_sha256']" in command
        assert "activation.ok" in command
        assert (
            "activation.payload['result']['package_sha256'] "
            "== selected['package_sha256']"
        ) in command
        for entrypoint in EXPECTED_CONSOLE_SCRIPTS:
            assert f'/{entrypoint}" --help' in command
    sdist_step = str(sdist_install_step["run"])
    install_index = sdist_step.index(
        "uv pip install --python .release-sdist-venv/bin/python"
    )
    chdir_index = sdist_step.index('cd "$sdist_smoke_root"')
    import_index = sdist_step.index(
        '"$GITHUB_WORKSPACE/.release-sdist-venv/bin/python" -c '
    )
    assert 'sdist_smoke_root="$(mktemp -d)"' in sdist_step
    assert '"${RELEASE_SDIST}[mcp]"' in sdist_step
    assert "--no-build-isolation" in sdist_step
    assert '"${RELEASE_WHEEL}[mcp]"' in str(wheel_install_step["run"])
    assert install_index < chdir_index < import_index
    upload_step = next(
        step
        for step in release_steps
        if str(step.get("uses") or "").startswith("actions/upload-artifact@")
    )
    assert "if" not in upload_step
    assert "continue-on-error" not in upload_step
    assert upload_step["with"]["path"].splitlines() == [
        "${{ env.RELEASE_WHEEL }}",
        "${{ env.RELEASE_SDIST }}",
        "${{ env.RELEASE_CHECKSUMS }}",
    ]
    assert upload_step["with"]["name"] == (
        "agent-libos-canonical-${{ github.sha }}-${{ github.run_attempt }}"
    )
    assert upload_step["with"]["retention-days"] == 14
