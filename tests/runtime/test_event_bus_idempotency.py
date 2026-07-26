from __future__ import annotations

import threading
from pathlib import Path

import pytest

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import EventPriority, EventType
from agent_libos.models.exceptions import ValidationError


class _DictSubclass(dict[str, object]):
    pass


@pytest.mark.parametrize("emitter_name", ["emit", "emit_once"])
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("event_type", 1, "event type"),
        ("source", 1, "event source"),
        ("target", False, "event target"),
        ("payload", [], "event payload"),
        ("payload", _DictSubclass(), "event payload"),
        ("priority", 1, "event priority"),
        ("correlation_id", False, "event correlation_id"),
        ("causality", [], "event causality"),
        ("causality", _DictSubclass(), "event causality"),
    ],
)
def test_emit_paths_share_exact_event_field_validation(
    emitter_name: str,
    field: str,
    value: object,
    message: str,
) -> None:
    runtime = Runtime.open("local")
    try:
        before = runtime.events.list()
        arguments: dict[str, object] = {
            "event_type": EventType.PROCESS_EXITED,
            "source": "pid_test",
        }
        arguments[field] = value
        if emitter_name == "emit_once":
            arguments["event_id"] = f"evt_test_invalid_{field}"

        with pytest.raises(ValidationError, match=message):
            getattr(runtime.events, emitter_name)(**arguments)

        assert runtime.events.list() == before
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    [
        ("event_type", EventType.PROCESS_CREATED),
        ("source", "different-source"),
        ("target", "different-target"),
        ("payload", {"pid": "different"}),
        ("priority", EventPriority.HIGH),
        ("correlation_id", "different-correlation"),
        ("causality", {"kind": "different"}),
    ],
)
def test_emit_once_rejects_every_semantic_field_collision(
    changed_field: str,
    changed_value: object,
) -> None:
    runtime = Runtime.open("local")
    event_id = f"evt_test_collision_{changed_field}"
    baseline: dict[str, object] = {
        "event_id": event_id,
        "event_type": EventType.PROCESS_EXITED,
        "source": "pid_test",
        "target": "pid_parent",
        "payload": {"pid": "pid_test", "status": "killed"},
        "priority": EventPriority.NORMAL,
        "correlation_id": "corr_test",
        "causality": {"kind": "terminal", "state_generation": 2},
    }
    try:
        original = runtime.events.emit_once(**baseline)
        conflicting = dict(baseline)
        conflicting[changed_field] = changed_value

        with pytest.raises(ValidationError, match="identity collision"):
            runtime.events.emit_once(**conflicting)

        assert runtime.events.store.get_event(event_id) == original
        assert len(
            [event for event in runtime.events.list() if event.event_id == event_id]
        ) == 1
    finally:
        runtime.close()
from agent_libos.runtime.event_bus import EventBus
from agent_libos.storage import SQLiteStore


@pytest.mark.parametrize(
    "invalid",
    [
        True,
        False,
        0,
        -1,
        1.5,
        "2",
        DEFAULT_CONFIG.gui.event_buffer_limit + 1,
    ],
)
def test_event_list_rejects_invalid_limit_before_store_query(
    invalid: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteStore(":memory:")
    events = EventBus(store)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(store, "list_events", lambda **kwargs: calls.append(kwargs))

    with pytest.raises(ValidationError, match="event list limit"):
        events.list(limit=invalid)  # type: ignore[arg-type]

    assert calls == []


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("target", 1),
        ("before_event_id", 1),
        ("after_event_id", 1),
        ("include_gui_presentation", 0),
        ("include_gui_presentation", "false"),
        ("include_gui_presentation", None),
    ],
)
def test_event_list_rejects_invalid_query_types_before_store_query(
    field: str,
    invalid: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteStore(":memory:")
    events = EventBus(store)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(store, "list_events", lambda **kwargs: calls.append(kwargs))

    with pytest.raises(ValidationError, match=field):
        events.list(**{field: invalid})  # type: ignore[arg-type]

    assert calls == []


def test_emit_once_is_concurrent_reopen_safe_and_collision_checked(
    tmp_path: Path,
) -> None:
    database = tmp_path / "idempotent-events.sqlite"
    event_id = "evt_test_stable_terminal_identity"
    runtime = Runtime.open(database)
    barrier = threading.Barrier(8)
    returned: list[object] = []
    errors: list[BaseException] = []

    def emit() -> None:
        try:
            with runtime.lifecycle.admit():
                barrier.wait(timeout=5)
                returned.append(
                    runtime.events.emit_once(
                        event_id,
                        EventType.PROCESS_EXITED,
                        source="pid_test",
                        target=None,
                        payload={"pid": "pid_test", "status": "killed"},
                        causality={
                            "kind": "process_terminal_state",
                            "pid": "pid_test",
                            "state_generation": 3,
                        },
                    )
                )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=emit) for _ in range(8)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert all(not thread.is_alive() for thread in threads)
        assert errors == []
        assert len(returned) == 8
        assert {event.event_id for event in returned} == {event_id}
        created_at = {event.created_at for event in returned}
        assert len(created_at) == 1
        assert len(
            [event for event in runtime.events.list() if event.event_id == event_id]
        ) == 1
        with pytest.raises(ValidationError, match="identity collision"):
            runtime.events.emit_once(
                event_id,
                EventType.PROCESS_EXITED,
                source="pid_test",
                target=None,
                payload={"pid": "pid_test", "status": "failed"},
                causality={
                    "kind": "process_terminal_state",
                    "pid": "pid_test",
                    "state_generation": 3,
                },
            )
    finally:
        for thread in threads:
            thread.join(timeout=5)
        runtime.close()

    reopened = Runtime.open(database)
    try:
        replayed = reopened.events.emit_once(
            event_id,
            EventType.PROCESS_EXITED,
            source="pid_test",
            target=None,
            payload={"pid": "pid_test", "status": "killed"},
            causality={
                "kind": "process_terminal_state",
                "pid": "pid_test",
                "state_generation": 3,
            },
        )
        assert replayed.event_id == event_id
        assert replayed.created_at in created_at
        assert len(
            [event for event in reopened.events.list() if event.event_id == event_id]
        ) == 1
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "error_type",
    [RuntimeError, KeyboardInterrupt],
    ids=["exception", "base-exception"],
)
def test_emit_once_link_failure_rolls_back_before_retry(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
) -> None:
    runtime = Runtime.open("local")
    event_id = f"evt_test_link_rollback_{error_type.__name__}"
    operations = runtime.events.operations
    assert operations is not None
    original_link = operations.link_evidence

    def fail_after_link(*args: object, **kwargs: object) -> object:
        result = original_link(*args, **kwargs)
        raise error_type("injected event evidence link failure")

    try:
        monkeypatch.setattr(operations, "link_evidence", fail_after_link)
        with pytest.raises(error_type):
            runtime.events.emit_once(
                event_id,
                EventType.PROCESS_EXITED,
                source="pid_test",
                payload={"pid": "pid_test", "status": "killed"},
            )

        assert runtime.events.store.get_event(event_id) is None
        monkeypatch.setattr(operations, "link_evidence", original_link)
        emitted = runtime.events.emit_once(
            event_id,
            EventType.PROCESS_EXITED,
            source="pid_test",
            payload={"pid": "pid_test", "status": "killed"},
        )
        assert emitted.event_id == event_id
        assert runtime.events.store.get_event(event_id) == emitted
    finally:
        runtime.close()
