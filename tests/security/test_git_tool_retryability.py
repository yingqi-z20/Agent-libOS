from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import GitErrorCode
from agent_libos.models.exceptions import GitError
from agent_libos.tools.base import ToolContext, ToolErrorCode, ToolResult
from agent_libos.tools.builtin.git import GitStageTool, GitStatusTool


class _FailingGitBoundary:
    def __init__(self, error: GitError) -> None:
        self._error = error

    def __getattr__(self, _name: str) -> Callable[..., None]:
        def fail(**_kwargs: object) -> None:
            raise self._error

        return fail


def _context(error: GitError) -> ToolContext:
    return ToolContext(
        trace_id="trace-git-retryability",
        call_id="call-git-retryability",
        pid="pid-git-retryability",
        runtime=SimpleNamespace(
            config=DEFAULT_CONFIG,
            git=_FailingGitBoundary(error),
        ),
    )


def _invoke_stage(error: GitError) -> ToolResult:
    return GitStageTool().invoke(
        {
            "paths": ["tracked.txt"],
            "expected_state_token": "0" * 64,
        },
        _context(error),
    )


def test_git_mutation_model_tool_does_not_mark_post_dispatch_stale_retryable() -> None:
    result = _invoke_stage(
        GitError(
            GitErrorCode.STALE_STATE.value,
            "repository identity changed after dispatch",
            operation="stage",
            retryable=True,
            details={"effect": "unknown"},
        )
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.TRANSIENT_ERROR
    assert result.error.retryable is False


def test_git_mutation_model_tool_never_marks_unknown_effect_retryable() -> None:
    result = _invoke_stage(
        GitError(
            GitErrorCode.UNKNOWN_EFFECT.value,
            "Git mutation timed out after dispatch",
            operation="stage",
            retryable=True,
            details={"effect": "unknown"},
        )
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.TRANSIENT_ERROR
    assert result.error.retryable is False


def test_git_mutation_model_tool_does_not_mark_ambiguous_timeout_retryable() -> None:
    result = _invoke_stage(
        GitError(
            GitErrorCode.TIMEOUT.value,
            "Git mutation timed out after dispatch",
            operation="stage",
            retryable=True,
            details={"effect": "unknown"},
        )
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.TIMEOUT
    assert result.error.retryable is False


def test_git_mutation_model_tool_keeps_initial_cas_stale_retryable_after_reobserve() -> None:
    result = _invoke_stage(
        GitError(
            GitErrorCode.STALE_STATE.value,
            "Git repository state changed before mutation dispatch",
            operation="stage",
            retryable=True,
            details={
                "actual_state_token": "1" * 64,
                "effect_started": False,
            },
        )
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.TRANSIENT_ERROR
    assert result.error.retryable is True


@pytest.mark.parametrize(
    "error",
    (
        GitError(
            GitErrorCode.STALE_STATE.value,
            "Git changed while it was being read",
            retryable=True,
        ),
        GitError(
            GitErrorCode.REPOSITORY_BUSY.value,
            "Git repository is busy",
            retryable=True,
        ),
    ),
    ids=("stale", "repository-busy"),
)
def test_git_read_model_tool_preserves_safe_transient_retryability(
    error: GitError,
) -> None:
    result = GitStatusTool().invoke({}, _context(error))

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.TRANSIENT_ERROR
    assert result.error.retryable is True


def test_git_model_tool_does_not_invent_repository_busy_retryability() -> None:
    result = GitStatusTool().invoke(
        {},
        _context(
            GitError(
                GitErrorCode.REPOSITORY_BUSY.value,
                "invalid repository lock timeout",
                retryable=False,
            )
        ),
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.TRANSIENT_ERROR
    assert result.error.retryable is False
