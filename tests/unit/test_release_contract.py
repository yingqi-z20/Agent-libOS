from __future__ import annotations

from pathlib import Path
import re
import tomllib

import yaml

from scripts.check_release_artifacts import validate_version_alignment


ROOT = Path(__file__).resolve().parents[2]


def test_release_version_identifiers_are_aligned() -> None:
    assert validate_version_alignment(ROOT) == "0.3.3"


def test_release_status_contains_current_version_state_only() -> None:
    text = (ROOT / "docs" / "release_status.md").read_text(encoding="utf-8")
    assert text.startswith("# Agent libOS 0.3.3 Status\n")
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


def test_python_wheel_scope_is_the_core_package() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["agent_libos"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "The Python wheel contains the core `agent_libos` package" in readme


def test_declared_python_support_has_an_explicit_upper_bound() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["requires-python"] == ">=3.11,<3.15"
    lock_header = (ROOT / "uv.lock").read_text(encoding="utf-8").splitlines()[:4]
    assert 'requires-python = ">=3.11, <3.15"' in lock_header


def test_console_entrypoint_uses_the_domain_error_boundary() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"] == {
        "agent-libos": "agent_libos.api.cli:cli",
        "agent-libos-gui-server": "agent_libos.api.gui.server:main",
    }


def test_ci_checkout_does_not_persist_credentials_in_git_config() -> None:
    workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text(
        encoding="utf-8"
    )
    parsed = yaml.safe_load(workflow)
    checkout_jobs: list[str] = []
    for job_name, job in parsed["jobs"].items():
        checkout_steps = [
            step
            for step in job.get("steps", [])
            if str(step.get("uses") or "").startswith("actions/checkout@")
        ]
        if not checkout_steps:
            continue
        checkout_jobs.append(job_name)
        assert len(checkout_steps) == 1
        assert checkout_steps[0].get("with", {}).get("persist-credentials") is False
    assert checkout_jobs


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


def test_release_workflow_preserves_and_clean_installs_validated_artifacts() -> None:
    workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text(
        encoding="utf-8"
    )
    parsed = yaml.safe_load(workflow)
    release_job = parsed["jobs"]["release-artifacts"]
    required_jobs = {
        "static",
        "python",
        "security",
        "deterministic-release",
        "postgres",
        "gui",
    }
    assert set(release_job["needs"]) == required_jobs
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
    for job_name in (
        "python",
        "security",
        "deterministic-release",
        "release-artifacts",
    ):
        deno_step = next(
            item
            for item in parsed["jobs"][job_name]["steps"]
            if item.get("name") == "Set up Deno"
        )
        assert deno_step["uses"] == "denoland/setup-deno@v2.0.4"
        assert deno_step["with"]["deno-version"] == "lts"
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
            "gui",
            "Run GUI checks",
            (
                "npm --prefix gui run test",
                "npm --prefix gui run typecheck",
                "npm --prefix gui run build",
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
    assert postgres_service["image"] == "postgres:17"
    assert postgres_service["ports"] == ["5432:5432"]
    postgres_step = next(
        item
        for item in postgres_job["steps"]
        if item.get("name") == "Run PostgreSQL store integration tests"
    )
    assert postgres_step["env"]["AGENT_LIBOS_POSTGRES_DSN"] == (
        "postgresql://agent_libos:agent_libos@127.0.0.1:5432/agent_libos"
    )
    release_steps = release_job["steps"]
    release_run_scripts = "\n".join(
        str(step.get("run") or "") for step in release_steps
    )
    build_step = next(
        step
        for step in release_steps
        if step.get("name") == "Build and validate distributions"
    )
    assert "if" not in build_step
    assert "continue-on-error" not in build_step
    assert "uv build --clear --out-dir dist" in str(build_step["run"])
    assert "python scripts/check_release_artifacts.py dist" in str(build_step["run"])
    assert release_run_scripts.count("uv pip check --python") == 2
    runtime_safety_step = next(
        str(step["run"])
        for step in parsed["jobs"]["deterministic-release"]["steps"]
        if step.get("name") == "Run runtime-safety release smoke"
    )
    assert "experiments/run_benchmark.py" in runtime_safety_step
    assert "--require-all-passed" in runtime_safety_step
    wheel_install_step = next(
        step
        for step in release_steps
        if step.get("name") == "Clean-install wheel and run entrypoint smoke"
    )
    sdist_install_step = next(
        step
        for step in release_steps
        if step.get("name") == "Clean-install source distribution"
    )
    for step in (wheel_install_step, sdist_install_step):
        assert "if" not in step
        assert "continue-on-error" not in step
    sdist_step = str(sdist_install_step["run"])
    install_index = sdist_step.index(
        "uv pip install --python .release-sdist-venv/bin/python"
    )
    chdir_index = sdist_step.index('cd "$sdist_smoke_root"')
    import_index = sdist_step.index(
        '"$GITHUB_WORKSPACE/.release-sdist-venv/bin/python" -c '
    )
    assert 'sdist_smoke_root="$(mktemp -d)"' in sdist_step
    assert install_index < chdir_index < import_index
    upload_step = next(
        step
        for step in release_steps
        if str(step.get("uses") or "").startswith("actions/upload-artifact@")
    )
    assert "if" not in upload_step
    assert "continue-on-error" not in upload_step
    assert upload_step["with"]["path"] == "dist/*"
