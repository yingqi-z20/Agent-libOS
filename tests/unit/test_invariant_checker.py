from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest import MonkeyPatch

from scripts import check_test_invariants as checker
from scripts import test_matrix


class TestInvariantChecker:

    def test_manifest_loader_accepts_yaml_syntax(self, tmp_path: Path) -> None:
        manifest = tmp_path / "invariants.yaml"
        manifest.write_text(
            """
schema_version: 1
invariants:
  - id: sample-invariant
    title: Sample invariant
    lane: unit
    node_ids:
      - tests/unit/test_sample.py::TestSample::test_regression
    benchmark_attack_classes:
      - sample_attack
benchmark_attack_classes:
  sample_attack: sample-invariant
""".lstrip(),
            encoding="utf-8",
        )

        data = checker._load_manifest(manifest)

        assert data["invariants"][0]["id"] == "sample-invariant"
        assert data["benchmark_attack_classes"]["sample_attack"] == "sample-invariant"

    def test_invariant_nodes_must_exist_and_include_deterministic_regression(self) -> None:
        manifest = {
            "schema_version": checker.MANIFEST_SCHEMA_VERSION,
            "invariants": [
                {
                    "id": "real-only",
                    "title": "Real only",
                    "lane": "security",
                    "node_ids": ["tests/security/test_real.py::TestReal::test_live"],
                    "benchmark_attack_classes": [],
                },
                {
                    "id": "missing-node",
                    "title": "Missing node",
                    "lane": "security",
                    "node_ids": ["tests/security/test_missing.py::TestMissing::test_absent"],
                    "benchmark_attack_classes": [],
                },
            ]
        }
        errors: list[str] = []

        checker._check_invariants(
            manifest,
            collected={"tests/security/test_real.py::TestReal::test_live"},
            deterministic_collected=set(),
            errors=errors,
        )

        assert any("real-only: requires at least one deterministic regression node" in error for error in errors)
        assert any("missing-node: pytest node not collected" in error for error in errors)

    def test_default_deterministic_collection_matches_the_test_matrix(self) -> None:
        args = SimpleNamespace(
            workers="1",
            dist="loadfile",
            durations=None,
            skip_real_deno=False,
            run_real_llm=False,
            run_mcp=False,
        )

        command = test_matrix._pytest_args(("tests",), args)

        assert command[-2:] == [
            "-m",
            checker.DEFAULT_DETERMINISTIC_MARKER_EXPRESSION,
        ]

    def test_main_collects_the_default_matrix_marker_expression(
        self,
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        collected_expressions: list[str | None] = []
        monkeypatch.setattr(checker, "_load_manifest", lambda _path: {})
        monkeypatch.setattr(
            checker,
            "_collect_pytest_nodeids",
            lambda expression=None: collected_expressions.append(expression) or set(),
        )
        monkeypatch.setattr(
            checker,
            "_check_invariants",
            lambda *_args, **_kwargs: (set(), {}),
        )
        monkeypatch.setattr(
            checker,
            "_check_benchmark_attack_classes",
            lambda *_args: None,
        )
        monkeypatch.setattr(
            checker,
            "_check_documented_invariants",
            lambda *_args: None,
        )

        status = checker.main(
            [
                "--manifest",
                str(tmp_path / "manifest.yaml"),
                "--documentation",
                str(tmp_path / "invariants.md"),
            ]
        )

        assert status == 0
        assert collected_expressions == [
            None,
            checker.DEFAULT_DETERMINISTIC_MARKER_EXPRESSION,
            (
                checker.DEFAULT_DETERMINISTIC_MARKER_EXPRESSION
                + " and platform_darwin"
            ),
            (
                checker.DEFAULT_DETERMINISTIC_MARKER_EXPRESSION
                + " and platform_linux"
            ),
        ]

    def test_optional_only_nodes_do_not_satisfy_deterministic_evidence(self) -> None:
        default_node = "tests/security/test_default.py::test_default"
        postgres_node = "tests/runtime/test_postgres.py::test_postgres"
        mcp_node = "tests/providers/test_mcp.py::test_mcp"
        manifest = {
            "schema_version": checker.MANIFEST_SCHEMA_VERSION,
            "invariants": [
                {
                    "id": "postgres-only",
                    "title": "PostgreSQL only",
                    "lane": "runtime",
                    "node_ids": [postgres_node],
                    "benchmark_attack_classes": [],
                },
                {
                    "id": "mcp-only",
                    "title": "MCP only",
                    "lane": "providers",
                    "node_ids": [mcp_node],
                    "benchmark_attack_classes": [],
                },
                {
                    "id": "default-backed",
                    "title": "Default backed",
                    "lane": "security",
                    "node_ids": [postgres_node, mcp_node, default_node],
                    "benchmark_attack_classes": [],
                },
            ]
        }
        errors: list[str] = []

        checker._check_invariants(
            manifest,
            collected={default_node, postgres_node, mcp_node},
            deterministic_collected={default_node},
            errors=errors,
        )

        assert errors == [
            "postgres-only: requires at least one deterministic regression node",
            "mcp-only: requires at least one deterministic regression node",
        ]

    def test_collected_but_skipped_node_does_not_satisfy_execution_evidence(self) -> None:
        node = "tests/security/test_runtime.py::test_invariant"
        manifest = {
            "invariants": [
                {
                    "id": "executed-invariant",
                    "title": "Executed invariant",
                    "lane": "security",
                    "node_ids": [node],
                }
            ]
        }
        errors: list[str] = []

        checker._check_invariant_execution(
            manifest,
            executed_nodeids=set(),
            errors=errors,
            lane="security",
            platform="linux",
            selected_nodeids={node},
        )

        assert errors == [
            "executed-invariant: no declared regression node completed without "
            "skip in the security lane"
        ]

    def test_passing_node_satisfies_execution_evidence(self) -> None:
        node = "tests/security/test_runtime.py::test_invariant"
        manifest = {
            "invariants": [
                {
                    "id": "executed-invariant",
                    "title": "Executed invariant",
                    "lane": "security",
                    "node_ids": [node],
                    "required_platform_nodes": {"linux": [node]},
                }
            ]
        }
        errors: list[str] = []

        checker._check_invariant_execution(
            manifest,
            executed_nodeids={node},
            errors=errors,
            lane="security",
            platform="linux",
        )

        assert errors == []

    def test_sharded_execution_checks_only_selected_test_files(self) -> None:
        first = "tests/runtime/test_first.py::test_first"
        second = "tests/runtime/test_second.py::test_second"
        manifest = {
            "invariants": [
                {
                    "id": "sharded-invariant",
                    "title": "Sharded invariant",
                    "lane": "runtime",
                    "node_ids": [first, second],
                    "required_platform_nodes": {"linux": [first, second]},
                }
            ]
        }
        errors: list[str] = []

        checker._check_invariant_execution(
            manifest,
            executed_nodeids={first},
            errors=errors,
            lane="runtime",
            platform="linux",
            selected_test_paths=("tests/runtime/test_first.py",),
        )

        assert errors == []

    def test_sharded_execution_ignores_marker_deselected_nodes(self) -> None:
        deterministic = "tests/providers/test_default.py::test_default"
        mcp_only = "tests/providers/test_mcp_sdk.py::test_sdk"
        manifest = {
            "invariants": [
                {
                    "id": "mixed-marker-invariant",
                    "title": "Mixed marker invariant",
                    "lane": "providers",
                    "node_ids": [deterministic, mcp_only],
                }
            ]
        }
        errors: list[str] = []

        checker._check_invariant_execution(
            manifest,
            executed_nodeids=set(),
            errors=errors,
            lane="providers",
            platform="windows",
            selected_test_paths=("tests/providers/test_mcp_sdk.py",),
            selected_nodeids=set(),
        )

        assert errors == []

    def test_directory_selected_paths_cover_descendant_nodes(self) -> None:
        node = "tests/runtime/test_nested.py::test_evidence"
        manifest = {
            "invariants": [
                {
                    "id": "directory-scoped-invariant",
                    "title": "Directory-scoped invariant",
                    "lane": "runtime",
                    "node_ids": [node],
                }
            ]
        }
        errors: list[str] = []

        checker._check_invariant_execution(
            manifest,
            executed_nodeids=set(),
            errors=errors,
            lane="runtime",
            platform="windows",
            selected_test_paths=("tests/runtime",),
            selected_nodeids={node},
        )

        assert errors == [
            "directory-scoped-invariant: no declared regression node completed "
            "without skip in the runtime lane"
        ]
        errors.clear()
        checker._check_invariant_execution(
            manifest,
            executed_nodeids={node},
            errors=errors,
            lane="runtime",
            platform="windows",
            selected_test_paths=("tests/runtime",),
            selected_nodeids={node},
        )
        assert errors == []

    def test_directory_selected_paths_still_enforce_required_platform_node(self) -> None:
        generic = "tests/security/test_monitor.py::test_generic"
        required = "tests/security/test_monitor.py::test_linux"
        manifest = {
            "invariants": [
                {
                    "id": "directory-platform-invariant",
                    "title": "Directory platform invariant",
                    "lane": "security",
                    "node_ids": [generic, required],
                    "required_platform_nodes": {"linux": [required]},
                }
            ]
        }
        errors: list[str] = []

        checker._check_invariant_execution(
            manifest,
            executed_nodeids={generic},
            errors=errors,
            lane="security",
            platform="linux",
            selected_test_paths=("tests/security",),
            selected_nodeids={generic, required},
        )

        assert errors == [
            "directory-platform-invariant: required linux pytest node did not "
            f"complete without skip: {required}"
        ]

    def test_platform_scoped_invariant_is_not_required_on_other_platform(
        self,
    ) -> None:
        darwin_node = "tests/security/test_identity.py::test_darwin"
        linux_node = "tests/security/test_identity.py::test_linux"
        manifest = {
            "invariants": [
                {
                    "id": "platform-identity",
                    "title": "Platform identity",
                    "lane": "security",
                    "node_ids": [darwin_node, linux_node],
                    "required_platform_nodes": {
                        "darwin": [darwin_node],
                        "linux": [linux_node],
                    },
                }
            ]
        }
        errors: list[str] = []

        checker._check_invariant_execution(
            manifest,
            executed_nodeids=set(),
            errors=errors,
            lane="security",
            platform="windows",
        )

        assert errors == []

    def test_generic_invariant_node_remains_required_on_other_platform(
        self,
    ) -> None:
        generic_node = "tests/security/test_identity.py::test_generic"
        darwin_node = "tests/security/test_identity.py::test_darwin"
        manifest = {
            "invariants": [
                {
                    "id": "mixed-platform-identity",
                    "title": "Mixed platform identity",
                    "lane": "security",
                    "node_ids": [generic_node, darwin_node],
                    "required_platform_nodes": {"darwin": [darwin_node]},
                }
            ]
        }
        errors: list[str] = []

        checker._check_invariant_execution(
            manifest,
            executed_nodeids=set(),
            errors=errors,
            lane="security",
            platform="windows",
        )

        assert errors == [
            "mixed-platform-identity: no declared regression node completed "
            "without skip in the security lane"
        ]

    def test_pytest_receipt_records_only_non_xfail_passing_calls(
        self,
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        receipt = tmp_path / "receipt.json"
        # This unit test invokes the live pytest plugin hooks directly.  Give
        # it an isolated receipt set so its synthetic session start cannot
        # erase nodes already recorded by the surrounding test-matrix run.
        monkeypatch.setattr(checker, "_EXECUTED_NODEIDS", set())
        monkeypatch.setenv(checker.INVARIANT_EXECUTION_RECEIPT_ENV, str(receipt))
        session = SimpleNamespace(config=SimpleNamespace())
        checker.pytest_sessionstart(session)

        checker.pytest_runtest_logreport(
            SimpleNamespace(
                nodeid="tests/unit/test_ok.py::test_ok",
                when="call",
                passed=True,
            )
        )
        checker.pytest_runtest_logreport(
            SimpleNamespace(
                nodeid="tests/unit/test_skip.py::test_skip",
                when="call",
                passed=False,
            )
        )
        checker.pytest_runtest_logreport(
            SimpleNamespace(
                nodeid="tests/unit/test_xfail.py::test_xfail",
                when="call",
                passed=True,
                wasxfail="known defect",
            )
        )
        checker.pytest_sessionfinish(session, 0)

        assert checker.load_execution_receipt(receipt) == {
            "tests/unit/test_ok.py::test_ok"
        }

    @pytest.mark.parametrize("schema_version", [True, 1, "2", None])
    def test_manifest_schema_version_must_be_exact(
        self,
        schema_version: object,
    ) -> None:
        node = "tests/security/test_default.py::test_default"
        manifest = {
            "schema_version": schema_version,
            "invariants": [
                {
                    "id": "default-backed",
                    "title": "Default backed",
                    "lane": "security",
                    "node_ids": [node],
                    "benchmark_attack_classes": [],
                }
            ],
        }
        errors: list[str] = []

        checker._check_invariants(
            manifest,
            collected={node},
            deterministic_collected={node},
            errors=errors,
        )

        assert errors == [
            "manifest schema_version must be exact integer "
            f"{checker.MANIFEST_SCHEMA_VERSION}, got {schema_version!r}"
        ]

    def test_required_platform_nodes_accept_exact_marker_collection(self) -> None:
        node = "tests/security/test_platform.py::test_darwin"
        manifest = {
            "schema_version": checker.MANIFEST_SCHEMA_VERSION,
            "invariants": [
                {
                    "id": "platform-backed",
                    "title": "Platform backed",
                    "lane": "security",
                    "node_ids": [node],
                    "required_platform_nodes": {"darwin": [node]},
                    "benchmark_attack_classes": [],
                }
            ],
        }
        errors: list[str] = []

        checker._check_invariants(
            manifest,
            collected={node},
            deterministic_collected={node},
            errors=errors,
            platform_collected={"darwin": {node}, "linux": set()},
        )

        assert errors == []

    @pytest.mark.parametrize(
        ("required_platform_nodes", "node_ids", "platform_collected", "expected"),
        [
            pytest.param(
                {"windows": ["tests/security/test_platform.py::test_default"]},
                ["tests/security/test_platform.py::test_default"],
                {},
                "required platform must be one of",
                id="unknown-platform",
            ),
            pytest.param(
                {"darwin": []},
                ["tests/security/test_platform.py::test_default"],
                {"darwin": set()},
                "must be a non-empty list",
                id="empty-platform-nodes",
            ),
            pytest.param(
                {"darwin": ["tests/security/test_platform.py::test_other"]},
                ["tests/security/test_platform.py::test_default"],
                {"darwin": {"tests/security/test_platform.py::test_other"}},
                "is not declared in node_ids",
                id="not-an-invariant-node",
            ),
            pytest.param(
                {"darwin": ["tests/security/test_platform.py::test_default"]},
                ["tests/security/test_platform.py::test_default"],
                {"darwin": set()},
                "is not deterministically collected with platform_darwin",
                id="missing-platform-marker",
            ),
        ],
    )
    def test_required_platform_nodes_fail_closed(
        self,
        required_platform_nodes: dict[str, list[str]],
        node_ids: list[str],
        platform_collected: dict[str, set[str]],
        expected: str,
    ) -> None:
        manifest = {
            "schema_version": checker.MANIFEST_SCHEMA_VERSION,
            "invariants": [
                {
                    "id": "platform-backed",
                    "title": "Platform backed",
                    "lane": "security",
                    "node_ids": node_ids,
                    "required_platform_nodes": required_platform_nodes,
                    "benchmark_attack_classes": [],
                }
            ],
        }
        collected = set(node_ids)
        errors: list[str] = []

        checker._check_invariants(
            manifest,
            collected=collected,
            deterministic_collected=collected,
            errors=errors,
            platform_collected=platform_collected,
        )

        assert any(expected in error for error in errors)

    def test_benchmark_attack_class_mapping_must_match_declarations_and_tasks(self, monkeypatch: MonkeyPatch) -> None:
        manifest = {
            "benchmark_attack_classes": {
                "declared_elsewhere": "other-invariant",
                "undeclared": "known-invariant",
                "unknown_owner": "missing-invariant",
            }
        }
        monkeypatch.setattr(
            checker,
            "load_tasks",
            lambda _suite: [
                SimpleNamespace(attack_class="task_without_mapping", source_path=None, id="task-1")
            ],
        )
        errors: list[str] = []

        checker._check_benchmark_attack_classes(
            manifest,
            invariant_ids={"known-invariant", "other-invariant"},
            declared_attack_classes={
                "declared_elsewhere": "known-invariant",
                "missing_top_level": "known-invariant",
            },
            errors=errors,
        )

        assert any("maps to 'other-invariant' but is declared on 'known-invariant'" in error for error in errors)
        assert any("'undeclared' is missing from invariant declarations" in error for error in errors)
        assert any("'unknown_owner' maps to unknown invariant 'missing-invariant'" in error for error in errors)
        assert any("'missing_top_level' is declared on 'known-invariant'" in error for error in errors)
        assert any("task_without_mapping" in error for error in errors)

    def test_documented_invariants_must_match_manifest_ids(self, tmp_path: Path) -> None:
        documentation = tmp_path / "invariants.md"
        documentation.write_text(
            """
## Current Invariant Groups

- `documented`: Current claim.
- `stale`: Removed claim.
- `documented`: Accidental duplicate.
""".lstrip(),
            encoding="utf-8",
        )
        manifest = {
            "invariants": [
                {"id": "documented"},
                {"id": "missing"},
            ]
        }
        errors: list[str] = []

        checker._check_documented_invariants(manifest, documentation, errors)

        assert any("duplicate invariant ids: documented" in error for error in errors)
        assert any("missing manifest invariant ids: missing" in error for error in errors)
        assert any("unknown invariant ids: stale" in error for error in errors)

    def test_repository_invariant_documentation_matches_manifest(self) -> None:
        errors: list[str] = []

        checker._check_documented_invariants(
            checker._load_manifest(checker.MANIFEST),
            checker.DOCUMENTATION,
            errors,
        )

        assert errors == []
