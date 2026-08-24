from __future__ import annotations

import pytest

from agent_libos.runtime.builder import RuntimeBuilder
from agent_libos.runtime.runtime import Runtime
from agent_libos.storage import SQLiteStore


def test_sync_open_rejects_custom_init_before_opening_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_calls = 0
    open_calls = 0

    class CustomInitRuntime(Runtime):
        def __init__(self, *args: object, **kwargs: object) -> None:
            nonlocal init_calls
            init_calls += 1
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            raise RuntimeError("constructor failed after assembly")

    def unexpected_open_store(*_args: object, **_kwargs: object) -> SQLiteStore:
        nonlocal open_calls
        open_calls += 1
        return SQLiteStore(":memory:")

    monkeypatch.setattr(
        "agent_libos.runtime.builder.open_store",
        unexpected_open_store,
    )

    with pytest.raises(
        TypeError,
        match=r"overrides Runtime\.__init__.*override allocate_unassembled",
    ):
        RuntimeBuilder.configured(CustomInitRuntime).open("local")

    assert init_calls == 0
    assert open_calls == 0


def test_sync_from_store_rejects_custom_init_without_owning_caller_store() -> None:
    init_calls = 0

    class CustomInitRuntime(Runtime):
        def __init__(self, *args: object, **kwargs: object) -> None:
            nonlocal init_calls
            init_calls += 1
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    store = SQLiteStore(":memory:")
    try:
        with pytest.raises(
            TypeError,
            match=r"overrides Runtime\.__init__.*override allocate_unassembled",
        ):
            RuntimeBuilder.configured(CustomInitRuntime).from_store(store)

        assert init_calls == 0
        assert store.list_processes() == []
    finally:
        store.close()


def test_sync_builder_uses_subclass_allocation_hook_without_calling_init() -> None:
    init_calls = 0

    class HookedRuntime(Runtime):
        def __init__(self, *args: object, **kwargs: object) -> None:
            nonlocal init_calls
            init_calls += 1
            raise AssertionError("builder must not call the custom constructor")

        @classmethod
        def allocate_unassembled(cls) -> Runtime:
            host = super().allocate_unassembled()
            host.allocated_by_sync_hook = True
            return host

    store = SQLiteStore(":memory:")
    runtime = RuntimeBuilder.configured(HookedRuntime).from_store(store)
    try:
        assert isinstance(runtime, HookedRuntime)
        assert runtime.allocated_by_sync_hook is True
        assert init_calls == 0
    finally:
        runtime.close()
