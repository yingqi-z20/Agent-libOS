from __future__ import annotations

from pathlib import PurePath
from typing import Sequence

from agent_libos.models.exceptions import ValidationError


# This is the complete historical direct shell:git compatibility surface.
# The argv policy also unwraps transparent executable launchers, but it cannot
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

_GIT_EXECUTABLE_LAUNCHERS = frozenset(
    {"env", "nice", "nohup", "setsid", "stdbuf", "timeout"}
)
_ENV_OPTIONS_WITH_VALUE = frozenset(
    {
        "-C",
        "--chdir",
        "-u",
        "--unset",
        "-a",
        "--argv0",
    }
)
_ENV_SPLIT_STRING_OPTIONS = frozenset({"-S", "--split-string"})
_TIMEOUT_OPTIONS_WITH_VALUE = frozenset(
    {"-k", "--kill-after", "-s", "--signal"}
)
_TIMEOUT_OPTIONS_WITHOUT_VALUE = frozenset(
    {
        "--foreground",
        "--preserve-status",
        "-v",
        "--verbose",
        "--help",
        "--version",
    }
)
_NICE_OPTIONS_WITH_VALUE = frozenset({"-n", "--adjustment"})
_NICE_OPTIONS_WITHOUT_VALUE = frozenset({"--help", "--version"})
_SETSID_OPTIONS_WITHOUT_VALUE = frozenset(
    {
        "-c",
        "--ctty",
        "-f",
        "--fork",
        "-w",
        "--wait",
        "--help",
        "--version",
    }
)
_STDBUF_OPTIONS_WITH_VALUE = frozenset(
    {"-i", "--input", "-o", "--output", "-e", "--error"}
)
_STDBUF_OPTIONS_WITHOUT_VALUE = frozenset({"--help", "--version"})

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


def _env_command(argv: Sequence[str]) -> list[str]:
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == "--":
            index += 1
            break
        option, separator, _value = token.partition("=")
        if option in _ENV_SPLIT_STRING_OPTIONS or token.startswith("-S"):
            raise ValidationError(
                "env split-string dispatch is unavailable; invoke the executable directly"
            )
        if token in _ENV_OPTIONS_WITH_VALUE:
            index += 2
            continue
        if separator and option in _ENV_OPTIONS_WITH_VALUE:
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        name, assignment, _value = token.partition("=")
        if assignment and name:
            index += 1
            continue
        break
    return list(argv[index:])


def _option_launcher_command(
    argv: Sequence[str],
    *,
    launcher: str,
    options_with_value: frozenset[str],
    options_without_value: frozenset[str],
    compact_value_prefixes: tuple[str, ...] = (),
    numeric_short_option: bool = False,
    positional_prefixes: int = 0,
) -> list[str]:
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == "--":
            index += 1
            break
        option, separator, _value = token.partition("=")
        if token in options_with_value:
            if index + 1 >= len(argv):
                raise ValidationError(f"{launcher} option requires a value")
            index += 2
            continue
        if separator and option in options_with_value:
            index += 1
            continue
        if any(
            token.startswith(prefix) and token != prefix
            for prefix in compact_value_prefixes
        ):
            index += 1
            continue
        if numeric_short_option and token.startswith("-") and token[1:].isdigit():
            index += 1
            continue
        if token in options_without_value:
            index += 1
            continue
        if token.startswith("-"):
            raise ValidationError(
                f"{launcher} option dispatch is unavailable; invoke the executable directly"
            )
        break
    if len(argv) - index < positional_prefixes:
        return []
    return list(argv[index + positional_prefixes :])


def _timeout_command(argv: Sequence[str]) -> list[str]:
    return _option_launcher_command(
        argv,
        launcher="timeout",
        options_with_value=_TIMEOUT_OPTIONS_WITH_VALUE,
        options_without_value=_TIMEOUT_OPTIONS_WITHOUT_VALUE,
        compact_value_prefixes=("-k", "-s"),
        positional_prefixes=1,
    )


def _nice_command(argv: Sequence[str]) -> list[str]:
    # POSIX and GNU nice both accept the historical ``-N`` adjustment form.
    return _option_launcher_command(
        argv,
        launcher="nice",
        options_with_value=_NICE_OPTIONS_WITH_VALUE,
        options_without_value=_NICE_OPTIONS_WITHOUT_VALUE,
        compact_value_prefixes=("-n",),
        numeric_short_option=True,
    )


def _setsid_command(argv: Sequence[str]) -> list[str]:
    return _option_launcher_command(
        argv,
        launcher="setsid",
        options_with_value=frozenset(),
        options_without_value=_SETSID_OPTIONS_WITHOUT_VALUE,
    )


def _stdbuf_command(argv: Sequence[str]) -> list[str]:
    return _option_launcher_command(
        argv,
        launcher="stdbuf",
        options_with_value=_STDBUF_OPTIONS_WITH_VALUE,
        options_without_value=_STDBUF_OPTIONS_WITHOUT_VALUE,
        compact_value_prefixes=("-i", "-o", "-e"),
    )


def _launcher_command(argv: Sequence[str]) -> list[str] | None:
    if not argv:
        return None
    launcher = _executable_name(argv[0])
    if launcher not in _GIT_EXECUTABLE_LAUNCHERS:
        return None
    if launcher == "env":
        return _env_command(argv)
    if launcher == "timeout":
        return _timeout_command(argv)
    if launcher == "nice":
        return _nice_command(argv)
    if launcher == "setsid":
        return _setsid_command(argv)
    if launcher == "stdbuf":
        return _stdbuf_command(argv)
    index = 2 if len(argv) > 1 and argv[1] == "--" else 1
    return list(argv[index:])


def _is_launcher_wrapped_git(argv: Sequence[str]) -> bool:
    nested = _launcher_command(argv)
    while nested is not None and nested:
        if is_git_invocation(nested):
            return True
        nested = _launcher_command(nested)
    return False


def validate_and_normalize_raw_git(argv: Sequence[str]) -> list[str]:
    """Normalize and restrict direct or transparently wrapped Git argv."""

    checked = list(argv)
    if _is_launcher_wrapped_git(checked):
        raise ValidationError(
            "raw Git mutation, remote, and arbitrary argv are unavailable; use a typed git_* tool"
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
