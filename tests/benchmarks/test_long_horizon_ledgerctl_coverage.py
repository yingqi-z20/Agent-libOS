"""Platform-independent checks of static coverage against unittest discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.long_horizon_agent.ledgerctl_scenario import (
    FIXTURE_FILES,
    regression_coverage,
)
from tests.benchmarks.test_long_horizon_ledgerctl import (
    _CLASS_DICT_STYLE_TESTS,
    _REGRESSION_TESTS,
    _apply_reference_fix,
    _run,
)


def _reference_workspace(tmp_path: Path) -> Path:
    # These checks need the fixture and child-process unittest execution, but
    # neither Git nor the platform-dependent bounded Host oracle.
    root = tmp_path / "workspace"
    for relative, source in FIXTURE_FILES.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
    _apply_reference_fix(root)
    return root


def _assert_suite_count(root: Path, expected: int) -> None:
    suite = _run(root, "-m", "unittest", "discover", "-s", "tests", "-q")
    assert suite.returncode == 0, suite.stderr
    assert f"Ran {expected} tests" in suite.stderr


@pytest.mark.parametrize(
    ("directory", "existing_packages"),
    [
        ("rounding", ()),
        ("rounding/nested", ("rounding",)),
        ("rounding/nested", ("rounding/nested",)),
    ],
)
def test_coverage_requires_every_discovered_package_initializer(
    tmp_path: Path, directory: str, existing_packages: tuple[str, ...],
) -> None:
    root = _reference_workspace(tmp_path)
    package = root / "tests" / directory
    package.mkdir(parents=True)
    for relative in existing_packages:
        (root / "tests" / relative / "__init__.py").touch()
    (root / "tests/test_rounding_consumers.py").rename(
        package / "test_rounding_consumers.py"
    )

    _assert_suite_count(root, 12)
    assert regression_coverage(root) == {
        "whole_unit_per_consumer": False,
        "negative_half_unit": False,
    }

    for parent in (package, *package.parents):
        if parent == root / "tests":
            break
        (parent / "__init__.py").touch()
    _assert_suite_count(root, 17)
    assert regression_coverage(root) == {
        "whole_unit_per_consumer": True,
        "negative_half_unit": True,
    }


def test_coverage_ignores_module_names_skipped_by_unittest(tmp_path: Path) -> None:
    root = _reference_workspace(tmp_path)
    (root / "tests/test_rounding_consumers.py").rename(
        root / "tests/test-rounding-consumers.py"
    )

    _assert_suite_count(root, 12)
    assert regression_coverage(root) == {
        "whole_unit_per_consumer": False,
        "negative_half_unit": False,
    }


@pytest.mark.parametrize("declaration", [
    "class RoundingConsumerTests:",
    "class TestCase:\n    pass\n\nclass RoundingConsumerTests(TestCase):",
    "from unittest import TestCase\nTestCase = object\n\nclass RoundingConsumerTests(TestCase):",
])
def test_coverage_ignores_methods_outside_discovered_testcase_classes(
    tmp_path: Path, declaration: str,
) -> None:
    root = _reference_workspace(tmp_path)
    source = _REGRESSION_TESTS.replace(
        "class RoundingConsumerTests(unittest.TestCase):", declaration,
    )
    (root / "tests/test_rounding_consumers.py").write_text(source, encoding="utf-8")

    _assert_suite_count(root, 12)
    assert regression_coverage(root) == {
        "whole_unit_per_consumer": False,
        "negative_half_unit": False,
    }


@pytest.mark.parametrize("declaration", [
    "import unittest as checks\n\nclass RoundingConsumerTests(checks.TestCase):",
    "from unittest import TestCase as Checks\n\nclass RoundingConsumerTests(Checks):",
    "from unittest.case import TestCase\n\nclass RoundingConsumerTests(TestCase):",
    "Checks = unittest.TestCase\n\nclass RoundingConsumerTests(Checks):",
    "class BaseChecks(unittest.TestCase):\n    pass\n\nclass RoundingConsumerTests(BaseChecks):",
])
def test_coverage_accepts_testcase_aliases_and_local_inheritance(
    tmp_path: Path, declaration: str,
) -> None:
    root = _reference_workspace(tmp_path)
    source = _REGRESSION_TESTS.replace(
        "class RoundingConsumerTests(unittest.TestCase):", declaration,
    )
    (root / "tests/test_rounding_consumers.py").write_text(source, encoding="utf-8")

    _assert_suite_count(root, 17)
    assert regression_coverage(root) == {
        "whole_unit_per_consumer": True,
        "negative_half_unit": True,
    }


@pytest.mark.parametrize("declaration", [
    "from test_support import BaseCase\n\nclass RoundingConsumerTests(BaseCase):",
    "import test_support as checks\n\nclass RoundingConsumerTests(checks.BaseCase):",
    "from supporting.base import BaseCase as Checks\n\nclass RoundingConsumerTests(Checks):",
])
def test_coverage_resolves_local_imported_testcase_ancestry_without_execution(
    tmp_path: Path, declaration: str,
) -> None:
    root = _reference_workspace(tmp_path)
    support = root / "tests/supporting"
    support.mkdir()
    (support / "__init__.py").touch()
    (support / "base.py").write_text(
        "from .ancestry import SupportCase\n"
        "class BaseCase(SupportCase):\n    pass\n",
        encoding="utf-8",
    )
    (support / "ancestry.py").write_text(
        "from pathlib import Path\n"
        "import unittest\n"
        "Path('support-imported').touch()\n"
        "class SupportCase(unittest.TestCase):\n    pass\n",
        encoding="utf-8",
    )
    (root / "tests/test_support.py").write_text(
        "from supporting.base import BaseCase\n", encoding="utf-8",
    )
    source = _REGRESSION_TESTS.replace(
        "class RoundingConsumerTests(unittest.TestCase):", declaration,
    )
    (root / "tests/test_rounding_consumers.py").write_text(source, encoding="utf-8")

    assert regression_coverage(root) == {
        "whole_unit_per_consumer": True,
        "negative_half_unit": True,
    }
    assert not (root / "support-imported").exists()
    _assert_suite_count(root, 17)
    assert (root / "support-imported").exists()


def test_coverage_resolves_partially_initialized_circular_support_imports(tmp_path: Path) -> None:
    root = _reference_workspace(tmp_path)
    (root / "tests/case_support.py").write_text(
        "import unittest\n"
        "class RootCase(unittest.TestCase):\n    pass\n"
        "from case_alias import ExportedCase\n"
        "class BaseCase(ExportedCase):\n    pass\n",
        encoding="utf-8",
    )
    (root / "tests/case_alias.py").write_text(
        "from case_support import RootCase as ExportedCase\n", encoding="utf-8",
    )
    source = _REGRESSION_TESTS.replace(
        "class RoundingConsumerTests(unittest.TestCase):",
        "from case_support import BaseCase\n\nclass RoundingConsumerTests(BaseCase):",
    )
    (root / "tests/test_rounding_consumers.py").write_text(source, encoding="utf-8")

    _assert_suite_count(root, 17)
    assert regression_coverage(root) == {
        "whole_unit_per_consumer": True,
        "negative_half_unit": True,
    }


@pytest.mark.parametrize("override_methods", [False, True])
def test_coverage_counts_mixin_methods_only_when_inherited_by_testcase(
    tmp_path: Path, override_methods: bool,
) -> None:
    root = _reference_workspace(tmp_path)
    source = _CLASS_DICT_STYLE_TESTS.replace(
        "class WholeUnitStyleTests(unittest.TestCase):", "class WholeUnitStyleTests:",
    )
    # Keep the test data and methods in a plain mixin; unittest discovers them
    # only through the module-bound TestCase subclass that inherits the mixin.
    assert "class WholeUnitStyleTests:" in source
    source += "\nclass DiscoveredTests(WholeUnitStyleTests, unittest.TestCase):\n"
    if override_methods:
        for name in (
            "test_validator_via_class_dict_unpacking",
            "test_exporter_via_fixture_config",
            "test_reporter_via_env_override",
        ):
            source += f"    def {name}(self):\n        self.assertTrue(True)\n"
    else:
        source += "    pass\n"
    (root / "tests/test_rounding_consumers.py").write_text(source, encoding="utf-8")

    _assert_suite_count(root, 15)
    assert regression_coverage(root) == {
        "whole_unit_per_consumer": not override_methods,
        "negative_half_unit": not override_methods,
    }


def test_coverage_preserves_class_dict_fixture_and_environment_styles(tmp_path: Path) -> None:
    root = _reference_workspace(tmp_path)
    (root / "tests/test_rounding_consumers.py").write_text(
        _CLASS_DICT_STYLE_TESTS, encoding="utf-8"
    )

    _assert_suite_count(root, 15)
    assert regression_coverage(root) == {
        "whole_unit_per_consumer": True,
        "negative_half_unit": True,
    }


@pytest.mark.parametrize("annotation", ["", ": LedgerConfig"])
def test_coverage_accepts_instance_configs_initialized_by_setup(
    tmp_path: Path, annotation: str,
) -> None:
    root = _reference_workspace(tmp_path)
    source = _REGRESSION_TESTS.replace(
        'JPY_UP = LedgerConfig(currency="JPY", precision=0, rounding="ROUND_HALF_UP")\n',
        "",
    ).replace(
        'JPY_EVEN = LedgerConfig(currency="JPY", precision=0, rounding="ROUND_HALF_EVEN")\n',
        "",
    )
    source = source.replace("JPY_UP", "self.JPY_UP").replace("JPY_EVEN", "self.JPY_EVEN")
    source = source.replace(
        "class RoundingConsumerTests(unittest.TestCase):\n",
        "class RoundingConsumerTests(unittest.TestCase):\n"
        "    def setUp(self):\n"
        f'        self.JPY_UP{annotation} = LedgerConfig(currency="JPY", precision=0, rounding="ROUND_HALF_UP")\n'
        f'        self.JPY_EVEN{annotation} = LedgerConfig(currency="JPY", precision=0, rounding="ROUND_HALF_EVEN")\n',
    )
    (root / "tests/test_rounding_consumers.py").write_text(source, encoding="utf-8")

    _assert_suite_count(root, 17)
    assert regression_coverage(root) == {
        "whole_unit_per_consumer": True,
        "negative_half_unit": True,
    }


@pytest.mark.parametrize("use_setup", [False, True])
def test_precision_bindings_do_not_leak_between_test_classes(
    tmp_path: Path, use_setup: bool,
) -> None:
    root = _reference_workspace(tmp_path)

    def configuration(precision: int) -> str:
        assignment = (
            f'config = LedgerConfig(currency="JPY", precision={precision}, '
            'rounding="ROUND_HALF_UP")\n'
        )
        return (
            "    def setUp(self):\n        self." + assignment
            if use_setup else "    " + assignment
        )

    source = '''import unittest
from decimal import Decimal
from ledgerctl.config import LedgerConfig
from ledgerctl.exporter import export_rows
from ledgerctl.reporter import summarize
from ledgerctl.validator import check_balance

def rows(*amounts):
    return [{"date": "d", "account": "a", "amount": amount} for amount in amounts]

class UnrelatedWholeUnitTests(unittest.TestCase):
''' + configuration(0) + '''
    def test_config(self):
        self.assertEqual(self.config.precision, 0)

class FractionalUnitTests(unittest.TestCase):
''' + configuration(2) + '''
    def test_exporter(self):
        self.assertEqual(export_rows(rows("-2.5"), self.config)[0]["amount"], "-2.50")
    def test_reporter(self):
        self.assertEqual(summarize(rows("-2.5"), self.config).total, Decimal("-2.50"))
    def test_validator(self):
        self.assertEqual(check_balance(rows("-2.5"), self.config), ["ledger does not balance: -2.50 JPY"])
'''
    (root / "tests/test_rounding_consumers.py").write_text(source, encoding="utf-8")

    _assert_suite_count(root, 16)
    assert regression_coverage(root) == {
        "whole_unit_per_consumer": False,
        "negative_half_unit": False,
    }
