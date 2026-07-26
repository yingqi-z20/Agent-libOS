from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import re
import tarfile
import tomllib
import zipfile

import pytest
import yaml

from agent_libos.skills.builtin_catalog import (
    BUILTIN_SKILL_IDS as RUNTIME_BUILTIN_SKILL_IDS,
    BUILTIN_SKILL_MAX_FILE_BYTES as RUNTIME_BUILTIN_SKILL_MAX_FILE_BYTES,
    BUILTIN_SKILL_MAX_INSTRUCTION_BYTES as RUNTIME_BUILTIN_SKILL_MAX_INSTRUCTION_BYTES,
)
from scripts.check_release_artifacts import (
    ALLOWED_SECRET_FIXTURE_SHA256,
    BUILTIN_SKILL_ARCHIVE_PATHS,
    BUILTIN_SKILL_IDS,
    CHECKSUM_MANIFEST_NAME,
    EXPECTED_CONSOLE_SCRIPTS,
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


def _wheel_metadata(version: str) -> bytes:
    return (
        "Metadata-Version: 2.4\n"
        "Name: agent-libos\n"
        f"Version: {version}\n"
        "Requires-Python: <3.15,>=3.11\n"
        "\n"
    ).encode()


def _write_test_wheel(
    target: Path,
    *,
    version: str = "1.0.0",
    wheel_metadata: bytes | None = None,
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
        archive.writestr(f"{dist_info}/METADATA", _wheel_metadata(version))
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
) -> Path:
    prefix = f"agent_libos-{version}"
    payloads: dict[str, bytes] = {}
    for relative in SDIST_REQUIRED_FILES:
        if relative == "PKG-INFO":
            payloads[relative] = _wheel_metadata(version)
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


def _write_release_pair(target: Path, *, version: str = "1.0.0") -> tuple[Path, Path]:
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
    assert validate_version_alignment(ROOT) == "1.0.0"


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
        "mcp": ["mcp>=1.27,<2"],
        "pty": ["pywinpty>=2.0.13,<3; sys_platform == 'win32'"],
    }
    assert pyproject["dependency-groups"]["release"] == [
        "check-wheel-contents==0.6.3",
        "hatchling==1.31.0",
        "twine==6.2.0",
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
    assert text.startswith("# Agent libOS 1.0.0 Status\n")
    forbidden = {
        "commit id": r"\bcommit\b",
        "dirty state": r"\bdirty\b",
        "worktree state": r"\bwork(?:ing)?[ -]?tree\b",
        "content hash": r"\bsha(?:-?256)?\b",
        "benchmark artifact path": r"\.benchmark_runs/",
        "absolute user path": r"(?:/Users/|/home/|/private/|/tmp/|[A-Za-z]:\\Users\\)",
        "bare hexadecimal identifier": r"\b[0-9a-f]{7,40}\b",
        "calendar date": r"\b20\d{2}-\d{2}-\d{2}\b",
    }
    offenders = [label for label, pattern in forbidden.items() if re.search(pattern, text, re.IGNORECASE)]
    assert offenders == []


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
    assert "not Agent libOS 1.0.0 release evidence" in text
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
    assert "The GUI lane passes the complete checked-in Vitest suite" in normalized
    assert (
        "Exact file and test counts are intentionally left to the CI receipt"
        in normalized
    )
    assert not re.search(r"\b\d+ Vitest files\b", text)
    assert not re.search(r"\b\d+ tests\b", text)


def test_documented_no_skip_provider_gates_invoke_pytest_directly() -> None:
    text = (ROOT / "docs" / "development.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "tests/providers/test_mcp_sdk_integration.py" in text
    assert "tests/self_evolution/test_builtin_agent_images_real_llm.py" in text
    assert not re.search(r"scripts/test_matrix\.py[^\n]*--fail-on-skip", text)
    assert "`--fail-on-skip` is a pytest option" in normalized


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

    assert "dist/agent_libos-1.0.0-py3-none-any.whl" in readme
    assert "dist/agent_libos-1.0.0.tar.gz" in readme
    assert readme.count("--require-hashes") >= 3
    assert "--no-deps dist/agent_libos-1.0.0-py3-none-any.whl" in readme
    assert "--no-deps --no-build-isolation dist/agent_libos-1.0.0.tar.gz" in readme
    for entrypoint in EXPECTED_CONSOLE_SCRIPTS:
        assert readme.count(f"/{entrypoint} --help") >= 2
    assert readme.count("uv pip check --python /tmp/agent-libos-") >= 2


def test_declared_python_support_has_an_explicit_upper_bound() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["requires-python"] == ">=3.11,<3.15"
    lock_header = (ROOT / "uv.lock").read_text(encoding="utf-8").splitlines()[:4]
    assert 'requires-python = ">=3.11, <3.15"' in lock_header


def test_gui_runtime_engines_are_explicit_and_lockfile_aligned() -> None:
    package = json.loads((ROOT / "gui" / "package.json").read_text(encoding="utf-8"))
    lockfile = json.loads(
        (ROOT / "gui" / "package-lock.json").read_text(encoding="utf-8")
    )

    assert package["engines"] == {
        "node": ">=22.12.0",
        "npm": ">=8",
    }
    assert lockfile["packages"][""]["engines"] == package["engines"]
    development = " ".join(
        (ROOT / "docs" / "development.md").read_text(encoding="utf-8").split()
    )
    support = " ".join(
        (ROOT / "docs" / "support_matrix.md").read_text(encoding="utf-8").split()
    )
    assert "GUI package declares Node `>=22.12.0` with npm 8 or newer" in development
    assert "CI exercises Node 24 with the npm version supplied" in development
    assert "lower compatibility bounds are not separate per-change CI jobs" in support
    assert "^20.19.0" not in development


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


def test_release_docs_distinguish_windows_ci_from_remaining_environment_gates(
) -> None:
    status = " ".join(
        (ROOT / "docs" / "release_status.md").read_text(encoding="utf-8").split()
    )
    support = " ".join(
        (ROOT / "docs" / "support_matrix.md").read_text(encoding="utf-8").split()
    )

    for required in (
        "the complete deterministic `all` lane on Windows 3.11",
        "checked-in CI coverage, not a separate local Windows run",
        (
            "there is no Windows Job Object parent-death containment or "
            "wall/CPU/RSS supervisor"
        ),
        "Deterministic local Git path/locking tests run in Windows CI",
    ):
        assert required in status
    for required in (
        "The checked-in Windows 3.11 job is CI evidence",
        "it is not a claim of a separate local Windows run",
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
        "git status --porcelain=v1 --untracked-files=all",
        "git submodule status --recursive",
        "git lfs ls-files --all --long",
        "git --no-replace-objects rev-list --objects --all",
        "git --no-replace-objects cat-file",
        "git --no-replace-objects cat-file --batch-all-objects",
        "set -o pipefail",
        'ANON_GIT_SCAN_DIR="$(mktemp -d)" || exit 1',
        'test -n "$ANON_GIT_SCAN_DIR" || exit 1',
        'test -n "${ANON_GIT_SCAN_DIR:-}" && test -d "$ANON_GIT_SCAN_DIR" || exit 1',
        'case "$object_type" in',
        'commit|tag|tree|blob)',
        "rg -a -qi",
        'ANON_COMMIT_TREE="$ANON_GIT_SCAN_DIR/exact-commit-tree"',
        'ANON_COMMIT_EXPORT="$ANON_GIT_SCAN_DIR/archive-projection"',
        'git --no-replace-objects archive --format=tar "$ANON_COMMIT"',
        '"$ANON_COMMIT_TREE" "$ANON_COMMIT_EXPORT" "$ANON_OUTPUT_DIR"; do',
        "find \"$ANON_BINARY_ROOT\" -type f -exec file --mime-type",
        "-iname '*.whl'",
        "Never copy a live secret",
    ):
        assert required in text
    assert text.count('commit|tag|tree|blob)') >= 2
    assert text.count(
        'git --no-replace-objects cat-file "$object_type" "$object_oid"'
    ) >= 2
    assert "--batch-check='%(objectname) %(objecttype) %(objectsize) %(rest)'" in text
    assert "structural package check is not an anonymity scan" in normalized
    assert (
        "raw exact-commit tree, deliverable archive projection, and "
        "generated-output inventory"
    ) in normalized
    assert "`export-ignore` can omit tracked paths" in normalized
    assert "`export-subst` can rewrite bytes" in normalized
    assert '["git", "--no-replace-objects", "cat-file", "blob", oid]' in text


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
        "runtime",
        "self-evolution",
        "providers",
        "benchmark",
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
    assert windows_install["run"] == "uv sync --frozen --all-groups --extra pty"
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
    assert host_identity_install["run"] == "uv sync --frozen --all-groups"
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
    for job_name in (
        "python",
        "security",
        "deterministic-release",
        "windows",
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
            "Run complete MCP SDK integration file",
            (
                "tests/providers/test_mcp_sdk_integration.py",
                "--run-mcp --fail-on-skip",
            ),
        ),
        (
            "gui",
            "Run GUI checks",
            (
                "npm --prefix gui run test",
                "npm --prefix gui run typecheck",
                "npm --prefix gui run build",
            ),
        ),
        (
            "windows",
            "Run complete deterministic Python matrix",
            (
                "scripts/test_matrix.py --lane all",
                "--max-lane-seconds 900",
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
    for job_name, step_name in (
        ("python", "Run pytest lane"),
        ("security", "Run complete security lane"),
    ):
        step = next(
            item
            for item in parsed["jobs"][job_name]["steps"]
            if item.get("name") == step_name
        )
        assert step["timeout-minutes"] == 15
        command = str(step["run"])
        assert "--durations 25" in command
        assert "--max-lane-seconds 360" in command
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
    mcp_install = next(
        item
        for item in mcp_job["steps"]
        if item.get("name") == "Install frozen MCP SDK dependencies"
    )
    assert mcp_install["run"] == "uv sync --frozen --all-groups --extra mcp"
    mcp_test = next(
        item
        for item in mcp_job["steps"]
        if item.get("name") == "Run complete MCP SDK integration file"
    )
    mcp_command = str(mcp_test["run"])
    assert "-k " not in mcp_command
    assert "--ignore" not in mcp_command
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
        "RELEASE_WHEEL": "dist/agent_libos-1.0.0-py3-none-any.whl",
        "RELEASE_SDIST": "dist/agent_libos-1.0.0.tar.gz",
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
    assert "uv export --frozen --no-dev --no-emit-project" in export_command
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
        assert "get_builtin_skill_catalog" in command
        assert "len(catalog.list()) == 26" in command
        assert "len(owned) == 99" in command
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
    assert '"$RELEASE_SDIST"' in sdist_step
    assert "--no-build-isolation" in sdist_step
    assert '"$RELEASE_WHEEL"' in str(wheel_install_step["run"])
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
