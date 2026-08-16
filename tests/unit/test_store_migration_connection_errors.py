from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from typing import Any, Callable

import pytest

import agent_libos.api.cli as cli_module
import agent_libos.storage.mcp_v7_migration as mcp_v7_migration
import agent_libos.storage.semantic_v5_migration as semantic_v5_migration
import agent_libos.storage.semantic_v6_migration as semantic_v6_migration
from agent_libos.models.exceptions import ValidationError


_RAW_HOST = "raw-migration-host-sentinel.invalid"
_RAW_PASSWORD = "raw-password-sentinel"
_RAW_USER = "raw-user-sentinel"
_RAW_PORT = "65431"
_RAW_DSN = (
    f"postgresql://{_RAW_USER}:{_RAW_PASSWORD}@{_RAW_HOST}:"
    f"{_RAW_PORT}/runtime"
)
_PLAN_SHA256 = "0" * 64


@dataclass(frozen=True)
class _MigrationCase:
    version: int
    module: Any
    plan: Callable[..., Any]
    apply: Callable[..., Any]
    error_type: type[ValidationError]


_CASES = (
    _MigrationCase(
        version=5,
        module=semantic_v5_migration,
        plan=semantic_v5_migration.plan_store_v5_migration,
        apply=semantic_v5_migration.apply_store_v5_migration,
        error_type=semantic_v5_migration.StoreV5MigrationError,
    ),
    _MigrationCase(
        version=6,
        module=semantic_v6_migration,
        plan=semantic_v6_migration.plan_store_v6_migration,
        apply=semantic_v6_migration.apply_store_v6_migration,
        error_type=semantic_v6_migration.StoreV6MigrationError,
    ),
    _MigrationCase(
        version=7,
        module=mcp_v7_migration,
        plan=mcp_v7_migration.plan_store_v7_migration,
        apply=mcp_v7_migration.apply_store_v7_migration,
        error_type=mcp_v7_migration.StoreV7MigrationError,
    ),
)


class _LeakingConnectionFailure:
    def __init__(self, dsn: str):
        raise RuntimeError(
            "provider connection failed for "
            f"{dsn}; host={_RAW_HOST}; port={_RAW_PORT}; "
            f"password={_RAW_PASSWORD}"
        )


def _case_id(case: _MigrationCase) -> str:
    return f"v{case.version}"


def _assert_no_connection_secret(value: str) -> None:
    for secret in (
        _RAW_DSN,
        _RAW_HOST,
        _RAW_PORT,
        _RAW_USER,
        _RAW_PASSWORD,
    ):
        assert secret not in value


def _assert_sanitized_error(
    error: BaseException,
    *,
    case: _MigrationCase,
) -> None:
    assert type(error) is case.error_type
    assert str(error) == (
        f"unable to open PostgreSQL schema-v{case.version} migration target"
    )
    assert error.__cause__ is None
    assert error.__context__ is None
    rendered = "".join(
        traceback.TracebackException.from_exception(error).format(chain=True)
    )
    serialized = json.dumps(
        {
            "type": type(error).__name__,
            "message": str(error),
            "repr": repr(error),
            "cause": error.__cause__,
            "context": error.__context__,
        },
        default=repr,
        sort_keys=True,
    )
    _assert_no_connection_secret(rendered)
    _assert_no_connection_secret(serialized)
    _assert_no_connection_secret(repr(vars(error)))


def _invoke(case: _MigrationCase, operation: str) -> Any:
    if operation == "plan":
        return case.plan(_RAW_DSN)
    return case.apply(
        _RAW_DSN,
        expected_plan_sha256=_PLAN_SHA256,
        postgres_snapshot_confirmed=True,
    )


def _cli_args(case: _MigrationCase, operation: str) -> list[str]:
    selected = [
        "--db",
        _RAW_DSN,
        "store",
        "migrate",
        "--to",
        str(case.version),
    ]
    if operation == "plan":
        return [*selected, "--dry-run"]
    return [
        *selected,
        "--apply",
        "--expected-plan-sha256",
        _PLAN_SHA256,
        "--postgres-snapshot-confirmed",
    ]


@pytest.mark.parametrize("case", _CASES, ids=_case_id)
@pytest.mark.parametrize("operation", ("plan", "apply"))
def test_postgres_migration_connection_failure_is_sanitized_at_api_and_cli(
    case: _MigrationCase,
    operation: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        case.module,
        "_PostgresConnection",
        _LeakingConnectionFailure,
    )

    with pytest.raises(case.error_type) as direct_failure:
        _invoke(case, operation)
    _assert_sanitized_error(direct_failure.value, case=case)

    with pytest.raises(case.error_type) as cli_failure:
        cli_module.main(_cli_args(case, operation))
    _assert_sanitized_error(cli_failure.value, case=case)
    captured = capsys.readouterr()
    _assert_no_connection_secret(captured.out)
    _assert_no_connection_secret(captured.err)
