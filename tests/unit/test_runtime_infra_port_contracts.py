from __future__ import annotations

import asyncio
import inspect
from concurrent.futures import Future
from typing import get_type_hints

import pytest

import agent_libos.ports.blocking_work as blocking_work
from agent_libos.ports.effects import ProtectedEffectPort
from agent_libos.ports.images import ImageFilesystemPort, ImageToolPort
from agent_libos.ports.messages import ProcessMessagePort
from agent_libos.ports.operations import OperationPort


class _TrackingExecutor:
    instances: list["_TrackingExecutor"] = []

    def __init__(self, **_kwargs: object) -> None:
        self.shutdown_calls: list[tuple[bool, bool]] = []
        self.__class__.instances.append(self)

    def submit(self, *_args: object, **_kwargs: object) -> Future[object]:
        raise RuntimeError("submit failed")

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        self.shutdown_calls.append((wait, cancel_futures))


def test_standalone_blocking_submit_failure_shuts_down_owned_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _TrackingExecutor.instances.clear()
    monkeypatch.setattr(blocking_work, "ThreadPoolExecutor", _TrackingExecutor)

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="submit failed"):
            await blocking_work.run_blocking_once(lambda: None)

    asyncio.run(exercise())

    assert len(_TrackingExecutor.instances) == 1
    assert _TrackingExecutor.instances[0].shutdown_calls == [(True, False)]


def test_standalone_blocking_wrap_failure_drains_submitted_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SubmittedExecutor(_TrackingExecutor):
        def submit(self, *_args: object, **_kwargs: object) -> Future[object]:
            future: Future[object] = Future()
            future.set_result(None)
            return future

    SubmittedExecutor.instances.clear()
    monkeypatch.setattr(blocking_work, "ThreadPoolExecutor", SubmittedExecutor)
    monkeypatch.setattr(
        asyncio,
        "wrap_future",
        lambda _future: (_ for _ in ()).throw(RuntimeError("wrap failed")),
    )

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="wrap failed"):
            await blocking_work.run_blocking_once(lambda: None)

    asyncio.run(exercise())

    assert SubmittedExecutor.instances[0].shutdown_calls == [(True, False)]


def test_effect_port_matches_keyword_and_result_contracts() -> None:
    insert = inspect.signature(ProtectedEffectPort.insert_external_effect)
    assert "record" in insert.parameters
    assert "effect" not in insert.parameters

    finalize = inspect.signature(ProtectedEffectPort.finalize_external_effect)
    assert list(finalize.parameters) == ["self", "intent_effect_id", "record"]
    assert get_type_hints(ProtectedEffectPort.finalize_external_effect)["return"] is bool

    transition = inspect.signature(ProtectedEffectPort.transition_external_effect)
    assert set(transition.parameters) == {
        "self",
        "effect_id",
        "expected_states",
        "transaction_state",
        "provider_metadata",
        "provider_receipt",
        "updated_at",
    }
    assert get_type_hints(ProtectedEffectPort.transition_external_effect)["return"] is bool
    assert get_type_hints(ProtectedEffectPort.abandon_external_effect_intent)["return"] is bool
    assert "cap_id" in inspect.signature(ProtectedEffectPort.get_capability).parameters


def test_image_filesystem_port_exposes_only_adapter_keywords_and_result_type() -> None:
    for method_name, expected in {
        "read_bytes": {"self", "pid", "path", "max_bytes", "cwd"},
        "read_directory": {"self", "pid", "path", "limit", "cwd"},
        "resolve_path": {"self", "path", "cwd"},
    }.items():
        signature = inspect.signature(getattr(ImageFilesystemPort, method_name))
        assert set(signature.parameters) == expected
        assert all(
            parameter.kind is not inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    return_type = get_type_hints(ImageFilesystemPort.resolve_path)["return"]
    resolved_type, display_type = return_type.__args__
    assert resolved_type.__name__ == "ResolvedPath"
    assert display_type is str


def test_image_tool_port_declares_required_initial_projection() -> None:
    signature = inspect.signature(ImageToolPort.initial_tool_projection)
    assert list(signature.parameters) == ["self", "image"]
    assert get_type_hints(ImageToolPort.initial_tool_projection)["return"] == list[str]


def test_message_and_operation_ports_declare_consumer_call_shape() -> None:
    notice = inspect.signature(ProcessMessagePort.notice)
    assert notice.parameters["instruction"].default is None
    assert get_type_hints(ProcessMessagePort.notice)["instruction"] == str | None

    wait = inspect.signature(OperationPort.wait)
    assert set(wait.parameters) == {"self", "operation_id", "metadata"}
    assert wait.parameters["operation_id"].kind is inspect.Parameter.KEYWORD_ONLY
