from __future__ import annotations

from pathlib import PurePath
from typing import Sequence

from agent_libos.models.exceptions import ValidationError


# This is the complete historical direct shell:git compatibility surface.
# Transparent executable launchers cannot safely preserve the identity and
# policy of the program they dispatch, so shared Shell/PTY argv validation
# rejects every launcher form that supplies a nested program.  It still cannot
# mediate Git started later by an authorized interpreter or native program.
READ_ONLY_GIT_COMMANDS: frozenset[tuple[str, ...]] = frozenset(
    {
        ("git", "status"),
        ("git", "status", "--short"),
        ("git", "branch", "--show-current"),
        ("git", "rev-parse", "--show-toplevel"),
        ("git", "diff"),
        ("git", "diff", "--stat"),
    }
)

_TRANSPARENT_EXECUTABLE_LAUNCHERS = frozenset(
    {"env", "nice", "nohup", "setsid", "stdbuf", "timeout"}
)
_READ_ONLY_OPERATION_BY_COMMAND: dict[tuple[str, ...], str] = {
    ("git", "status"): "status",
    ("git", "status", "--short"): "status",
    ("git", "branch", "--show-current"): "list_refs",
    ("git", "rev-parse", "--show-toplevel"): "repository_info",
    ("git", "diff"): "diff",
    ("git", "diff", "--stat"): "diff",
}


def _git_executable_name(value: str) -> str | None:
    raw = value.strip().replace("\\", "/")
    name = PurePath(raw).name.casefold()
    return "git" if name in {"git", "git.exe"} else None


def _executable_name(value: str) -> str:
    raw = value.strip().replace("\\", "/")
    name = PurePath(raw).name.casefold()
    return name[:-4] if name.endswith(".exe") else name


def is_git_invocation(argv: Sequence[str]) -> bool:
    return bool(argv) and _git_executable_name(argv[0]) == "git"


def validate_and_normalize_raw_git(argv: Sequence[str]) -> list[str]:
    """Apply shared launcher rejection and normalize exact legacy Git reads."""

    checked = list(argv)
    if (
        len(checked) > 1
        and _executable_name(checked[0]) in _TRANSPARENT_EXECUTABLE_LAUNCHERS
    ):
        raise ValidationError(
            "transparent executable launcher dispatch is unavailable; invoke the executable directly"
        )
    if not is_git_invocation(checked):
        return checked
    if "/" in checked[0] or "\\" in checked[0]:
        raise ValidationError(
            "raw Git executable paths are unavailable; use a typed git_* tool"
        )
    normalized = ["git", *checked[1:]]
    if tuple(normalized) not in READ_ONLY_GIT_COMMANDS:
        raise ValidationError(
            "raw Git mutation, remote, and arbitrary argv are unavailable; use a typed git_* tool"
        )
    return normalized


def harden_read_only_git_argv(argv: Sequence[str]) -> list[str]:
    """Return the shared Shell/PTY dispatch argv for an exact legacy Git read."""

    normalized = validate_and_normalize_raw_git(argv)
    if tuple(normalized) not in READ_ONLY_GIT_COMMANDS:
        return normalized
    return harden_trusted_git_read(normalized)


def harden_trusted_git_read(argv: Sequence[str]) -> list[str]:
    """Harden a Host-owned, already allowlisted Git read command."""

    checked = list(argv)
    if not is_git_invocation(checked) or len(checked) < 2:
        raise ValidationError("trusted Git read argv must start with a Git subcommand")
    hardened = [
        "git",
        "--no-pager",
        "--no-optional-locks",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-c",
        "maintenance.auto=false",
        "-c",
        "submodule.recurse=false",
        "-c",
        "diff.external=",
        "-c",
        "color.ui=false",
        checked[1],
    ]
    if checked[1] == "diff":
        hardened.extend(["--no-ext-diff", "--no-textconv"])
    hardened.extend(checked[2:])
    return hardened


def trusted_git_read_operation(
    argv: Sequence[str],
    *,
    hardened_only: bool = False,
) -> str | None:
    """Identify an exact legacy read before or after shared hardening.

    Host providers use this to run repository/config validation immediately
    before process dispatch.  Comparing against generated hardened argv keeps
    this recognizer closed over the same six-command compatibility surface.
    """

    checked = list(argv)
    if not is_git_invocation(checked):
        return None
    normalized = ["git", *checked[1:]]
    operation = _READ_ONLY_OPERATION_BY_COMMAND.get(tuple(normalized))
    if operation is not None and not hardened_only:
        return operation
    for command, selected_operation in _READ_ONLY_OPERATION_BY_COMMAND.items():
        if normalized == harden_trusted_git_read(command):
            return selected_operation
    return None


__all__ = [
    "READ_ONLY_GIT_COMMANDS",
    "harden_read_only_git_argv",
    "harden_trusted_git_read",
    "is_git_invocation",
    "trusted_git_read_operation",
    "validate_and_normalize_raw_git",
]
