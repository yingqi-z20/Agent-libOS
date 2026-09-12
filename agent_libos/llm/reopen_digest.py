"""Payload-free digest of a process's durable activity before a Runtime reopen.

Tool-result payloads live in runtime memory and are released when the Runtime
closes.  After a reopen the model sees only opaque omitted identifiers and, in
observed runs, re-read every file it had already read or written and repeated
work it could not see.  The durable event log still records *what* the process
did (paths read and written, commands run with return codes, Skills activated,
checkpoints created) without any tool output.  This module turns those events
into a bounded digest so the model re-observes only what it still needs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agent_libos.models import Event, EventType

REOPEN_DIGEST_HEADING = (
    "Durable activity before the last Runtime reopen "
    "(payload-free, from runtime events):"
)
REOPEN_DIGEST_GUIDANCE = (
    "- guidance: these are historical operations; later changes may have replaced "
    "or deleted their results. Re-read only the files you must edit or "
    "verify next, and use the Git inspection tools to see your own earlier edits "
    "instead of re-reading every file."
)

_LOST_OMISSION_REASONS = frozenset({"capability_denied", "missing"})
_DIGEST_EVENT_TYPES = frozenset(
    {
        EventType.EXTERNAL_READ.value,
        EventType.EXTERNAL_WRITE.value,
        EventType.SKILL_LOADED.value,
        EventType.CHECKPOINT_CREATED.value,
        EventType.HUMAN_OUTPUT.value,
    }
)
_DEFAULT_ITEMS_PER_LINE = 40
_COMPACT_ITEMS_PER_LINE = 12
_MAX_ARGV_CHARS = 120
_MAX_DIGEST_CHARS = 6_000


def context_lost_earlier_results(
    object_manifest: Sequence[Mapping[str, Any]] | None,
) -> bool:
    """True when the manifest omits earlier results whose payloads are gone."""

    for entry in object_manifest or ():
        if not isinstance(entry, Mapping) or entry.get("disposition") != "omitted":
            continue
        if str(entry.get("reason") or "") in _LOST_OMISSION_REASONS:
            return True
    return False


def collect_pre_reopen_events(
    list_events: Callable[..., list[Event]],
    pid: str,
    *,
    scan_limit: int,
    page_size: int,
) -> list[Event]:
    """Return this process's effect events older than the last Runtime shutdown.

    Effect events target the resource they touched rather than the process, so
    the store's target filter cannot select them.  Pages of the global log are
    walked backwards (newest first) until ``scan_limit`` events were seen or
    the log is exhausted.  Without a recorded shutdown nothing was lost and the
    result is empty; a partially scanned log yields a partial digest, never a
    wrong one, because only events before the shutdown are reported.
    """

    if scan_limit <= 0 or page_size <= 0:
        return []
    collected: list[Event] = []
    before: str | None = None
    scanned = 0
    while scanned < scan_limit:
        requested = min(page_size, scan_limit - scanned)
        page = list(list_events(limit=requested, before_event_id=before))
        if not page:
            break
        scanned += len(page)
        collected = page + collected
        before = page[0].event_id
        if len(page) < requested:
            break
    cutoff: str | None = None
    for event in reversed(collected):
        if _event_type(event) == EventType.RUNTIME_SHUTDOWN.value:
            cutoff = event.created_at
            break
    if cutoff is None:
        return []
    return [
        event
        for event in collected
        if event.created_at < cutoff
        and (event.source == pid or event.target == pid)
        and _event_type(event) in _DIGEST_EVENT_TYPES
    ]


def render_reopen_activity_digest(
    events: Sequence[Event],
    *,
    redact: Callable[[Any], Any] | None = None,
    on_included_event: Callable[[Event], None] | None = None,
) -> str:
    """Render activity and report only events that contribute digest facts."""

    activity = _summarize(events, redact=redact, on_included_event=on_included_event)
    text = _render(activity, items_per_line=_DEFAULT_ITEMS_PER_LINE)
    if len(text) > _MAX_DIGEST_CHARS:
        text = _render(activity, items_per_line=_COMPACT_ITEMS_PER_LINE)
    return text


def _event_type(event: Event) -> str:
    event_type = event.type
    if isinstance(event_type, EventType):
        return event_type.value
    return str(event_type or "")


class _Activity:
    """Mutable accumulator for one digest; rendered once collection ends."""

    def __init__(self) -> None:
        self.reads: list[str] = []
        self.directories: list[str] = []
        self.writes: dict[str, tuple[int, int]] = {}
        self.files_deleted: list[str] = []
        self.directories_ensured: list[str] = []
        self.directories_deleted: list[str] = []
        self.commands: list[tuple[str, Any, int]] = []
        self.git_operations: dict[str, int] = {}
        self.skills: list[str] = []
        self.checkpoints = 0
        self.human_outputs = 0

    def as_dict(self) -> dict[str, Any]:
        return dict(vars(self))


def _summarize(
    events: Sequence[Event],
    *,
    redact: Callable[[Any], Any] | None,
    on_included_event: Callable[[Event], None] | None,
) -> dict[str, Any]:
    activity = _Activity()
    for event in events:
        payload = redact(event.payload) if redact is not None else event.payload
        if not isinstance(payload, Mapping):
            continue
        handler = _EVENT_HANDLERS.get(_event_type(event))
        if handler is not None and handler(activity, payload):
            if on_included_event is not None:
                on_included_event(event)
    return activity.as_dict()


def _note_read(activity: _Activity, payload: Mapping[str, Any]) -> bool:
    adapter = payload.get("adapter")
    if adapter == "git":
        operation = str(payload.get("operation") or "inspect")
        activity.git_operations[operation] = activity.git_operations.get(operation, 0) + 1
        return True
    if adapter != "filesystem":
        return False
    path = payload.get("path")
    if not isinstance(path, str) or not path:
        return False
    target = (
        activity.directories
        if payload.get("operation") == "read_directory"
        else activity.reads
    )
    if path not in target:
        target.append(path)
    return True


def _note_write(activity: _Activity, payload: Mapping[str, Any]) -> bool:
    adapter = payload.get("adapter")
    if adapter == "filesystem":
        return _note_filesystem_write(activity, payload)
    elif adapter == "shell":
        argv = payload.get("argv")
        if isinstance(argv, list):
            _note_command(activity, argv, payload.get("returncode"))
            return True
    return False


def _note_filesystem_write(activity: _Activity, payload: Mapping[str, Any]) -> bool:
    path = payload.get("path")
    if not isinstance(path, str) or not path:
        return False
    operation = payload.get("operation")
    if operation is not None and not isinstance(operation, str):
        return False
    if operation in {"delete_file", "delete_directory"} and payload.get("deleted") is False:
        return False
    target = {
        "delete_file": activity.files_deleted,
        "write_directory": activity.directories_ensured,
        "delete_directory": activity.directories_deleted,
    }.get(operation)
    if target is not None:
        if path not in target:
            target.append(path)
        return True
    # write_text events historically omit operation, but include a byte count.
    # Unknown mutations must not be guessed to be successful file writes.
    written = payload.get("bytes_written")
    if operation not in {None, "write_text"} or type(written) is not int or written < 0:
        return False
    count, _ = activity.writes.get(path, (0, 0))
    activity.writes[path] = (count + 1, written)
    return True


def _note_command(activity: _Activity, argv: list[Any], returncode: Any) -> None:
    command = " ".join(str(part) for part in argv)
    if len(command) > _MAX_ARGV_CHARS:
        command = command[: _MAX_ARGV_CHARS - 3] + "..."
    if activity.commands and activity.commands[-1][0] == command:
        _, _, repeats = activity.commands[-1]
        activity.commands[-1] = (command, returncode, repeats + 1)
    else:
        activity.commands.append((command, returncode, 1))


def _note_skill(activity: _Activity, payload: Mapping[str, Any]) -> bool:
    skill_id = payload.get("skill_id")
    if not isinstance(skill_id, str) or not skill_id:
        return False
    if skill_id not in activity.skills:
        activity.skills.append(skill_id)
    return True


def _note_checkpoint(activity: _Activity, payload: Mapping[str, Any]) -> bool:
    activity.checkpoints += 1
    return True


def _note_human_output(activity: _Activity, payload: Mapping[str, Any]) -> bool:
    activity.human_outputs += 1
    return True


_EVENT_HANDLERS: dict[str, Callable[[_Activity, Mapping[str, Any]], bool]] = {
    EventType.EXTERNAL_READ.value: _note_read,
    EventType.EXTERNAL_WRITE.value: _note_write,
    EventType.SKILL_LOADED.value: _note_skill,
    EventType.CHECKPOINT_CREATED.value: _note_checkpoint,
    EventType.HUMAN_OUTPUT.value: _note_human_output,
}


def _join(items: Sequence[str], cap: int) -> str:
    shown = ", ".join(items[:cap])
    if len(items) > cap:
        shown += f", +{len(items) - cap} more"
    return shown


def _render(activity: Mapping[str, Any], *, items_per_line: int) -> str:
    lines: list[str] = []
    if activity["reads"]:
        lines.append("- files read: " + _join(activity["reads"], items_per_line))
    if activity["directories"]:
        lines.append(
            "- directories listed: " + _join(activity["directories"], items_per_line)
        )
    if activity["writes"]:
        entries = [
            f"{path} ({written} B{f' x{count}' if count > 1 else ''})"
            for path, (count, written) in activity["writes"].items()
        ]
        lines.append("- files written: " + _join(entries, items_per_line))
    for field, label in (
        ("files_deleted", "files deleted"),
        ("directories_ensured", "directories ensured"),
        ("directories_deleted", "directories deleted"),
    ):
        if activity[field]:
            lines.append(f"- {label}: " + _join(activity[field], items_per_line))
    if activity["commands"]:
        entries = [
            f"{command} -> returncode {returncode}{f' x{repeats}' if repeats > 1 else ''}"
            for command, returncode, repeats in activity["commands"]
        ]
        lines.append("- commands run: " + "; ".join(entries[:items_per_line]))
    if activity["git_operations"]:
        lines.append(
            "- git inspections: "
            + ", ".join(
                f"{operation} x{count}"
                for operation, count in sorted(activity["git_operations"].items())
            )
        )
    if activity["skills"]:
        lines.append("- skills activated: " + _join(activity["skills"], items_per_line))
    if activity["checkpoints"]:
        lines.append(f"- checkpoints created: {activity['checkpoints']}")
    if activity["human_outputs"]:
        lines.append(f"- human_output deliveries: {activity['human_outputs']}")
    if not lines:
        return ""
    return "\n".join([REOPEN_DIGEST_HEADING, *lines, REOPEN_DIGEST_GUIDANCE])
