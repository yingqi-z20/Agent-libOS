from __future__ import annotations

import json
from pathlib import Path
import re
import tomllib

from agent_libos.config.defaults import SkillDefaults
from scripts import test_matrix


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return ROOT.joinpath(path).read_text(encoding="utf-8")


def _words(text: str) -> str:
    return " ".join(text.split())


def _bash_blocks(document: str) -> list[str]:
    return re.findall(r"```bash\n(.*?)\n```", document, flags=re.DOTALL)


def test_release_runbook_matches_security_and_uses_bound_readback() -> None:
    security = _read("SECURITY.md")
    releasing = _read("docs/releasing.md")
    releasing_words = _words(releasing)

    assert "private vulnerability reporting is **enabled**" in security
    assert (
        "records that GitHub private vulnerability reporting is enabled"
        in releasing_words
    )
    assert (
        "reports that no confidential vulnerability intake is enabled"
        not in releasing_words
    )
    assert "Public publication is blocked if either check fails" in releasing_words

    assert '"${RELEASE_BASE_SHA:?set the exact reviewed base commit SHA}"' in releasing
    assert "scripts/check_changed_whitespace.py" in releasing
    assert '--base-sha "$RELEASE_BASE_SHA"' in releasing
    assert '--default-branch "$RELEASE_DEFAULT_BRANCH"' in releasing
    assert 'test "$RELEASE_BASE_SHA" != "$RELEASE_HEAD_SHA"' in releasing
    assert 'git merge-base --is-ancestor "$RELEASE_BASE_SHA" "$RELEASE_HEAD_SHA"' in releasing

    for generator in (
        "scripts/generate_cli_reference.py",
        "scripts/generate_config_reference.py",
        "scripts/generate_pypi_readme.py",
    ):
        write_command = f"uv run --frozen python {generator}"
        check_command = f"{write_command} --check"
        assert write_command in releasing.splitlines()
        assert check_command in releasing.splitlines()
        assert releasing.index(write_command) < releasing.index(check_command)

    for required in (
        "https://test.pypi.org/pypi",
        "https://pypi.org/pypi",
        "index file set differs",
        "index digest/size differs from canonical file",
        "downloaded file differs from canonical file",
        "long-description-links-ok",
        'f"/blob/v{version}/"',
        '"/blob/main/" in long_description',
        "metadata-and-entrypoints-ok",
        "entrypoint-help-and-demo-ok",
        '"agent-libos-gui-server"',
        '"agent-libos-migrate-tool-groups"',
    ):
        assert required in releasing

    python_heredocs = re.findall(r"<<'PY'\n(.*?)\nPY", releasing, flags=re.DOTALL)
    assert len(python_heredocs) >= 4
    for index, source in enumerate(python_heredocs):
        compile(source, f"docs/releasing.md heredoc {index}", "exec")


def test_standard_checks_distinguish_focused_and_complete_evidence() -> None:
    development = _read("docs/development.md")
    assert "## In this guide" in development
    assert "[Choose focused or complete checks](#standard-checks)" in development
    standard = development.split("## Standard Checks", 1)[1].split(
        "## Real LLM Smoke", 1
    )[0]

    assert "### Focused feedback (not complete)" in standard
    assert "starter set, not complete repository evidence" in standard
    assert "### Complete deterministic root-project baseline" in standard
    assert "uv run python scripts/test_matrix.py --lane all" in standard
    assert "npm --prefix gui ci" in standard
    assert "uv run python scripts/test_matrix.py --lane gui" in standard
    expected_lanes = (
        "unit",
        "runtime",
        "security",
        "self-evolution",
        "providers",
        "benchmark",
    )
    assert tuple(test_matrix.LANE_PATHS) == expected_lanes
    assert "all" in test_matrix.REAL_DENO_SERIAL_LANES
    assert "all" in test_matrix.DENO_RESOURCE_MONITOR_LANES
    for lane in expected_lanes:
        assert f"`{lane}`" in standard

    assert "The Python `all` lane is one command" not in development
    development_words = _words(development)
    assert "`all` can execute three commands sequentially" in development_words
    assert "not one aggregate timeout for the entire lane" in development_words


def test_desktop_docs_select_exact_cpython_and_do_not_repeat_nested_steps() -> None:
    package = json.loads(_read("gui/package.json"))
    distribution_script = package["scripts"]["desktop:dist"]
    assert "npm run desktop:stage" in distribution_script
    assert "check_desktop_artifacts.py" in distribution_script

    for path in ("docs/development.md", "docs/gui.md"):
        document = _read(path)
        blocks = [block for block in _bash_blocks(document) if "desktop:dist" in block]
        assert len(blocks) == 1, path
        block = blocks[0]
        for required in (
            "uv python install 3.11.15",
            "export UV_PYTHON=3.11.15",
            "export UV_MANAGED_PYTHON=1",
            'platform.python_implementation() == "CPython"',
            'platform.python_version() == "3.11.15"',
            "node --version | grep -Fx 'v24.15.0'",
            "npm --prefix gui run desktop:dist",
        ):
            assert required in block, (path, required)
        assert "desktop:stage" not in block
        assert "check_desktop_artifacts.py" not in block


def test_gui_has_a_scannable_first_task_user_path() -> None:
    gui = _words(_read("docs/gui.md"))

    for required in (
        "## In this guide",
        "[Complete a first task](#first-task-user-path)",
        "## First task: user path",
        "**Launch the app.**",
        "**Create the task.**",
        "**Follow execution.**",
        "**Handle input and permissions deliberately.**",
        "**Read the outcome.**",
        "launch are only request ceilings",
        "`failed`, `killed`, or Durable `needs_attention`",
        "**Audit** or **Explain**",
    ):
        assert required in gui


def test_browser_live_docs_gate_node_playwright_and_chromium_first() -> None:
    live_command = "experiments/run_browser_customer_flow_evaluation.py"
    for path in (
        "docs/benchmark.md",
        "benchmarks/browser_customer_workflows/README.md",
    ):
        document = _read(path)
        npm_ci = document.index("npm --prefix gui ci")
        chromium = document.index(
            "npm --prefix gui exec -- playwright install --with-deps chromium"
        )
        playwright = document.index("npm --prefix gui exec -- playwright --version")
        live = document.index(live_command)
        assert npm_ci < chromium < playwright < live, path
        assert "gui/package-lock.json" in document

    gui = _read("docs/gui.md")
    assert "npm --prefix gui ci" in gui
    assert "npm --prefix gui exec -- playwright install --with-deps chromium" in gui
    assert "npm --prefix gui run test:e2e" in gui


def test_sdist_examples_and_task_plan_metadata_state_their_boundaries() -> None:
    examples = _read("examples/mcp/README.md")
    examples_words = _words(examples)
    contributing = _words(_read("CONTRIBUTING.md"))
    pyproject = tomllib.loads(_read("pyproject.toml"))
    sdist = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]
    assert "/examples" in sdist["include"]
    assert "/uv.lock" in sdist["exclude"]
    assert "/gui" in sdist["exclude"]
    assert "reproducible only from a Git checkout" in examples_words
    assert "does not include that repository lock" in examples_words
    assert "uv pip install --python .venv-example/bin/python '.[mcp]'" in examples
    assert "not a frozen-lock reproduction or a release receipt" in examples_words
    assert "source distribution intentionally omits the repository `uv.lock`" in contributing
    assert "`uv sync` can resolve a new local environment" in contributing
    assert "not a frozen-lock reproduction or release receipt" in contributing
    assert "GUI sources are absent from the Python sdist" in contributing

    skills = _read("docs/skills.md")
    skills_words = _words(skills)
    assert SkillDefaults().resource_dirs == ("scripts", "references", "assets")
    for required in (
        "skills/task-plan/agents/openai.yaml",
        "not an Agent libOS Skill resource",
        "not readable through `read_skill_resource`",
        "does not contribute bytes to the Skill package SHA-256",
        "approval of the Agent libOS Skill hash does not authenticate it",
    ):
        assert required in skills_words
