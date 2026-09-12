"""Summarize one retained long-horizon Runtime database without copying prompts.

The long-horizon evaluation retains ``<artifacts>/run-N/state/runtime.sqlite``.
This inspector reads that database read-only and reports, per LLM call, the
timing, provider token usage, the character size of each top-level prompt
section, the tool schema size, and the tool calls the model returned.  It
prints sizes, counts, names, and categories only; prompt text, tool arguments,
model text, and provider payloads are never echoed. Close the owning Runtime
first; its exclusive SQLite lock prevents a concurrent consistent snapshot.

Example:

    uv run python experiments/inspect_long_horizon_run.py \
        .benchmark_runs/<name>-artifacts/run-1/state/runtime.sqlite --calls
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
from collections import Counter
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from agent_libos.llm.usage import canonicalize_llm_usage

# Top-level user-prompt headings emitted by ``agent_libos.llm.prompt``.  Text
# before the first heading is attributed to ``preamble``; unknown paragraph
# starts stay inside the current section.
PROMPT_SECTION_HEADINGS: tuple[tuple[str, str], ...] = (
    ("Available Skills (metadata only):", "available_skills"),
    ("Retained original goal contract", "original_goal"),
    ("Loaded skills:", "loaded_skills"),
    ("Compatibility tool schemas:", "compat_tool_schemas"),
    ("Materialized context:", "materialized_context"),
    ("LLM context object:", "llm_context_object"),
    ("Current runtime state (volatile", "runtime_state_heading"),
    ("Process facts:", "process_facts"),
    ("Materialized context metadata (volatile):", "context_metadata"),
    ("Materialized context warning:", "context_warning"),
    ("Durable activity before the last Runtime reopen", "reopen_digest"),
    ("Capabilities:", "capabilities"),
    ("Permission-request ceilings", "requestable_capabilities"),
    ("Recent events:", "recent_events"),
    ("Pending explicit process input", "pending_process_input"),
    ("The append-only LLM context object below", "llm_context_preamble"),
)

ACTION_CATEGORIES: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "skill_lifecycle",
        frozenset(
            {"discover_skills", "activate_skill", "read_skill_resource", "unload_skill"}
        ),
    ),
    (
        "terminal",
        frozenset({"process_exit", "human_output"}),
    ),
    (
        "verify",
        frozenset({"run_shell_command", "parse_pytest_log"}),
    ),
    (
        "mutate",
        frozenset(
            {"write_text_file", "write_directory", "delete_file", "delete_directory"}
        ),
    ),
    (
        "memory",
        frozenset(
            {
                "create_memory_object",
                "append_memory_object",
                "read_memory_object",
                "create_memory_namespace",
                "list_memory_namespace",
            }
        ),
    ),
    (
        "messages",
        frozenset({"read_process_messages", "receive_process_messages"}),
    ),
    (
        "checkpoint",
        frozenset({"create_checkpoint", "list_checkpoints", "inspect_checkpoint"}),
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Report per-call timing, token, prompt-section-size, and tool-call "
            "statistics for one retained long-horizon Runtime database."
        )
    )
    parser.add_argument("database", help="Path to a retained runtime.sqlite file.")
    parser.add_argument("--pid", help="Restrict to one process id.")
    parser.add_argument(
        "--calls",
        action="store_true",
        help="Print one line per LLM call in addition to the summary.",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        help="Also write the complete summary (with per-call rows) as JSON.",
    )
    args = parser.parse_args(argv)
    summary = inspect_database(Path(args.database), pid=args.pid)
    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(render_summary(summary, include_calls=args.calls))
    return 0


def inspect_database(path: Path, *, pid: str | None = None) -> dict[str, Any]:
    """Return a payload-free summary of the LLM calls persisted in ``path``."""

    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"database not found: {source}")
    with tempfile.TemporaryDirectory(prefix="agent-libos-inspect-") as scratch:
        # SQLite backup coordinates with WAL checkpoints and concurrent writers;
        # copying the main file and sidecars separately cannot preserve a snapshot.
        copied = Path(scratch) / source.name
        with (
            closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=0)) as live,
            closing(sqlite3.connect(copied)) as snapshot,
        ):
            # Runtime stores hold an exclusive SQLite lock. Refuse a locked
            # source instead of waiting forever inside backup's busy retry loop.
            def backup_progress(status: int, _remaining: int, _total: int) -> None:
                if status in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                    raise sqlite3.OperationalError(
                        "database is locked; close the owning Runtime before inspection"
                    )

            live.backup(snapshot, progress=backup_progress)
            return _inspect_connection(snapshot, pid=pid, source=str(source))


def _inspect_connection(
    connection: sqlite3.Connection, *, pid: str | None, source: str
) -> dict[str, Any]:
    connection.row_factory = sqlite3.Row
    query = "SELECT * FROM llm_calls"
    params: tuple[Any, ...] = ()
    if pid:
        query += " WHERE pid = ?"
        params = (pid,)
    query += " ORDER BY created_at, call_id"
    rows = list(connection.execute(query, params))
    calls = [_call_row(index, row) for index, row in enumerate(rows, start=1)]
    _attach_gaps(calls)
    pids = sorted({call["pid"] for call in calls if call["pid"]})
    # Audit activity can precede the first LLM call for the requested process.
    audit = _audit_summary(connection, [pid] if pid else [])
    objects = _object_summary(connection, pids)
    return {
        "source": source,
        "pids": pids,
        "calls": calls,
        "totals": _totals(calls),
        "sections": _section_totals(calls),
        "tool_calls": _tool_call_totals(calls),
        "audit": audit,
        "objects": objects,
    }


def _call_row(index: int, row: sqlite3.Row) -> dict[str, Any]:
    messages = _load_json(row["messages_json"], default=[])
    usage, _invalid_usage = canonicalize_llm_usage(
        _load_json(row["usage_json"], default={}), api=row["api"]
    )
    input_keys = (
        ("input_tokens", "prompt_tokens")
        if row["api"] == "responses"
        else ("prompt_tokens", "input_tokens")
    )
    output_keys = (
        ("output_tokens", "completion_tokens")
        if row["api"] == "responses"
        else ("completion_tokens", "output_tokens")
    )
    tool_calls = _load_json(row["tool_calls_json"], default=[])
    tools = row["tools_json"]
    created = _parse_time(row["created_at"])
    completed = _parse_time(row["completed_at"])
    sections: dict[str, int] = {}
    role_chars: dict[str, int] = {}
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "unknown")
            content = message.get("content")
            text = content if isinstance(content, str) else json.dumps(content)
            role_chars[role] = role_chars.get(role, 0) + len(text)
            if role == "user":
                for name, size in split_prompt_sections(text).items():
                    sections[name] = sections.get(name, 0) + size
    names = [_tool_call_name(item) for item in tool_calls] if isinstance(tool_calls, list) else []
    tool_specs = _load_json(tools, default=None)
    return {
        "index": index,
        "call_id": row["call_id"],
        "pid": row["pid"],
        "status": row["status"],
        "purpose": row["purpose"],
        "created_at": row["created_at"],
        "duration_s": (
            round((completed - created).total_seconds(), 1)
            if created and completed
            else None
        ),
        "gap_s": None,
        "input_tokens": _usage_int(usage, *input_keys),
        "cached_tokens": _usage_int(usage, "cache_read_tokens"),
        "output_tokens": _usage_int(usage, *output_keys),
        "reasoning_tokens": _usage_int(usage, "reasoning_tokens"),
        "role_chars": role_chars,
        "sections": sections,
        "tools_bytes": len(tools.encode("utf-8")) if isinstance(tools, str) else 0,
        "tool_count": len(tool_specs) if isinstance(tool_specs, list) else None,
        "tool_calls": [name for name in names if name],
        "response_chars": len(row["response_content"] or ""),
        "error_category": (
            _error_category(row["error"], _load_json(row["observability_json"], default={}))
            if row["status"] == "error"
            else None
        ),
    }


def split_prompt_sections(text: str) -> dict[str, int]:
    """Attribute user-prompt characters to the top-level runtime headings."""

    sizes: dict[str, int] = {}
    current = "preamble"
    for paragraph in text.split("\n\n"):
        first_line = paragraph.lstrip("\n").split("\n", 1)[0]
        for heading, name in PROMPT_SECTION_HEADINGS:
            if first_line.startswith(heading):
                current = name
                break
        sizes[current] = sizes.get(current, 0) + len(paragraph) + 2
    return sizes


def tool_call_category(name: str) -> str:
    """Return the coarse category (``observe``, ``mutate``, ...) for one tool name."""

    return _category(name)


def tool_call_name(item: Any) -> str:
    """Return the tool name from one persisted model tool-call entry, or ``""``."""

    return _tool_call_name(item)


def usage_int(usage: Any, *keys: str) -> int:
    """Read the first non-negative integer counter under ``keys`` (or nested details)."""

    return _usage_int(usage, *keys)


def parse_timestamp(value: Any) -> datetime | None:
    """Parse one persisted ISO-8601 timestamp, tolerating a trailing ``Z``."""

    return _parse_time(value)


def _attach_gaps(calls: list[dict[str, Any]]) -> None:
    previous_end: datetime | None = None
    for call in calls:
        started = _parse_time(call["created_at"])
        if started and previous_end:
            call["gap_s"] = round((started - previous_end).total_seconds(), 1)
        duration = call["duration_s"]
        if started is not None:
            previous_end = (
                started.__class__.fromtimestamp(
                    started.timestamp() + (duration or 0.0), tz=started.tzinfo
                )
                if duration is not None
                else started
            )


def _totals(calls: list[dict[str, Any]]) -> dict[str, Any]:
    first = _parse_time(calls[0]["created_at"]) if calls else None
    last_row = calls[-1] if calls else None
    last = _parse_time(last_row["created_at"]) if last_row else None
    if last is not None and last_row and last_row["duration_s"] is not None:
        last = last.__class__.fromtimestamp(
            last.timestamp() + last_row["duration_s"], tz=last.tzinfo
        )
    durations = [call["duration_s"] for call in calls if call["duration_s"] is not None]
    return {
        "llm_calls": len(calls),
        "error_calls": sum(call["status"] == "error" for call in calls),
        "wall_seconds": (
            round((last - first).total_seconds(), 1) if first and last else None
        ),
        "llm_seconds": round(sum(durations), 1),
        "max_call_seconds": max(durations) if durations else None,
        "input_tokens": sum(call["input_tokens"] for call in calls),
        "cached_tokens": sum(call["cached_tokens"] for call in calls),
        "output_tokens": sum(call["output_tokens"] for call in calls),
        "reasoning_tokens": sum(call["reasoning_tokens"] for call in calls),
        "user_prompt_chars": sum(call["role_chars"].get("user", 0) for call in calls),
        "system_prompt_chars": sum(
            call["role_chars"].get("system", 0) for call in calls
        ),
        "tools_bytes": sum(call["tools_bytes"] for call in calls),
        "multi_call_responses": sum(len(call["tool_calls"]) > 1 for call in calls),
        "empty_responses": sum(not call["tool_calls"] for call in calls),
    }


def _section_totals(calls: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {}
    for call in calls:
        for name, size in call["sections"].items():
            entry = totals.setdefault(name, {"chars": 0, "calls": 0, "max_chars": 0})
            entry["chars"] += size
            entry["calls"] += 1
            entry["max_chars"] = max(entry["max_chars"], size)
    return dict(sorted(totals.items(), key=lambda item: -item[1]["chars"]))


def _tool_call_totals(calls: list[dict[str, Any]]) -> dict[str, Any]:
    by_name: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    for call in calls:
        for name in call["tool_calls"]:
            by_name[name] += 1
            by_category[_category(name)] += 1
    return {
        "total": sum(by_name.values()),
        "by_name": dict(by_name.most_common()),
        "by_category": dict(by_category.most_common()),
    }


def _audit_summary(connection: sqlite3.Connection, pids: list[str]) -> dict[str, Any]:
    if not _table_exists(connection, "audit_records"):
        return {}
    rows = connection.execute(
        "SELECT actor, action, target, decision_json FROM audit_records"
    ).fetchall()
    process_targets = {f"process:{pid}" for pid in pids}
    selected = [
        row
        for row in rows
        if not pids
        or row["actor"] in pids
        or row["target"] in process_targets
    ]
    actions: Counter[str] = Counter(str(row["action"]) for row in selected)
    repair_tool_calls = 0
    tool_failures = 0
    for row in selected:
        decision = _load_json(row["decision_json"], default={})
        if row["action"] == "llm.action_repair_requested" and isinstance(decision, dict):
            count = decision.get("tool_call_count")
            repair_tool_calls += count if isinstance(count, int) and count > 0 else 1
        if row["action"] == "tool.call" and isinstance(decision, dict):
            if decision.get("ok") is False:
                tool_failures += 1
    return {
        "quanta": actions.get("scheduler.run_quantum", 0),
        "llm_requests": actions.get("llm.request", 0),
        "single_actions": actions.get("llm.action", 0),
        "action_batches": actions.get("llm.action_batch", 0),
        "action_repairs": actions.get("llm.action_repair_requested", 0),
        "repaired_tool_calls": repair_tool_calls,
        "tool_calls": actions.get("tool.call", 0),
        "tool_failures": tool_failures,
        "skill_activations": actions.get("skill.activate", 0),
        "context_materializations": actions.get("memory.materialize_context", 0),
        "exit_reviews_required": actions.get("process.exit_review_required", 0),
        "exit_reviews_passed": actions.get("process.exit_review_passed", 0),
        "human_messages_posted": actions.get("process.message.post", 0),
    }


def _object_summary(connection: sqlite3.Connection, pids: list[str]) -> dict[str, Any]:
    if not _table_exists(connection, "objects"):
        return {}
    rows = connection.execute(
        "SELECT type, created_by, length(payload_json) AS payload_chars FROM objects"
    ).fetchall()
    by_type: dict[str, dict[str, int]] = {}
    for row in rows:
        if pids and row["created_by"] not in pids:
            continue
        entry = by_type.setdefault(str(row["type"]), {"count": 0, "payload_chars": 0})
        entry["count"] += 1
        entry["payload_chars"] += int(row["payload_chars"] or 0)
    return dict(sorted(by_type.items()))


def render_summary(summary: dict[str, Any], *, include_calls: bool) -> str:
    lines: list[str] = []
    totals = summary["totals"]
    lines.append(f"database: {summary['source']}")
    lines.append(f"pids: {', '.join(summary['pids']) or '-'}")
    lines.append("")
    lines.append("totals:")
    for key, value in totals.items():
        lines.append(f"  {key}: {value}")
    if summary["audit"]:
        lines.append("")
        lines.append("audit:")
        for key, value in summary["audit"].items():
            lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("prompt sections (user role, chars summed over calls):")
    for name, entry in summary["sections"].items():
        mean = entry["chars"] // max(entry["calls"], 1)
        lines.append(
            f"  {name:28s} total={entry['chars']:>10,d} mean={mean:>8,d} "
            f"max={entry['max_chars']:>8,d} calls={entry['calls']}"
        )
    lines.append("")
    lines.append("tool calls returned by the model:")
    lines.append(f"  total: {summary['tool_calls']['total']}")
    for category, count in summary["tool_calls"]["by_category"].items():
        lines.append(f"  [{category}] {count}")
    for name, count in summary["tool_calls"]["by_name"].items():
        lines.append(f"    {name}: {count}")
    if summary["objects"]:
        lines.append("")
        lines.append("objects by type (count, persisted payload chars):")
        for name, entry in summary["objects"].items():
            lines.append(f"  {name}: {entry['count']} ({entry['payload_chars']:,d} chars)")
    if include_calls:
        lines.append("")
        lines.append(
            "calls: idx  start     dur    gap   input  cached   out  reason "
            "user_chars  ctx_chars  skills_chars  -> tool calls"
        )
        for call in summary["calls"]:
            sections = call["sections"]
            lines.append(
                f"  #{call['index']:>3d} {call['created_at'][11:19]} "
                f"{_fmt(call['duration_s']):>6s} {_fmt(call['gap_s']):>6s} "
                f"{call['input_tokens']:>7,d} {call['cached_tokens']:>7,d} "
                f"{call['output_tokens']:>5,d} {call['reasoning_tokens']:>6,d} "
                f"{call['role_chars'].get('user', 0):>10,d} "
                f"{sections.get('materialized_context', 0):>10,d} "
                f"{sections.get('loaded_skills', 0):>13,d}  -> "
                f"{', '.join(call['tool_calls']) or '(no tool call)'}"
                + (f"  [error:{call['error_category']}]" if call["error_category"] else "")
            )
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def _category(name: str) -> str:
    for category, names in ACTION_CATEGORIES:
        if name in names:
            return category
    if name.startswith(("read_", "get_", "list_", "inspect_", "git_")):
        return "observe"
    return "other"


def _tool_call_name(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    name = item.get("name")
    if isinstance(name, str):
        return name
    function = item.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    return ""


def _usage_int(usage: Any, *keys: str) -> int:
    if not isinstance(usage, dict):
        return 0
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    for nested_key in ("input_tokens_details", "prompt_tokens_details", "output_tokens_details", "completion_tokens_details"):
        nested = usage.get(nested_key)
        if isinstance(nested, dict):
            for key in keys:
                value = nested.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    return value
    return 0


def _error_category(error: Any, observability: Any = None) -> str:
    """Classify a sanitized provider failure without echoing its text.

    The persisted public error string is deliberately generic
    (``LLMTransientError``); the retained observability failure record carries
    the provider SDK exception type, which is what distinguishes a timeout from
    a rate limit or an HTTP status failure.
    """

    error_type = ""
    if isinstance(observability, dict):
        failure = observability.get("failure")
        if isinstance(failure, dict):
            internal = failure.get("internal_error")
            if isinstance(internal, dict):
                error_type = str(internal.get("error_type") or "")
    lowered_type = error_type.casefold()
    if "timeout" in lowered_type:
        return "timeout"
    if "ratelimit" in lowered_type or "rate_limit" in lowered_type:
        return "rate_limit"
    if "connection" in lowered_type:
        return "connection"
    if "status" in lowered_type:
        return "provider_http"
    message = str(error or "").casefold()
    if "timed out" in message or "timeout" in message:
        return "timeout"
    if "rate limit" in message or "status=429" in message:
        return "rate_limit"
    if any(marker in message for marker in ("connection", "dns", "tls")):
        return "connection"
    if "status=" in message:
        return "provider_http"
    return "provider_error"


def _load_json(value: Any, *, default: Any) -> Any:
    if not isinstance(value, str) or not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


if __name__ == "__main__":
    sys.exit(main())
