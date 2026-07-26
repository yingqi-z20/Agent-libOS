from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json

import pytest

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import AgentImage, ToolSpec
from agent_libos.models.exceptions import ValidationError
from agent_libos.modules.context import ModuleContext, ModuleRuntimeView
from agent_libos.modules.schema import ModuleManifest, ModuleProvides


class _CountingTool:
    def __init__(self, name: str, calls: Counter[str]) -> None:
        self.name = name
        self._calls = calls

    def spec(self) -> ToolSpec:
        self._calls[self.name] += 1
        return ToolSpec(name=self.name, description="module context complexity fixture")


def _context(*, declared_tools: tuple[str, ...]) -> ModuleContext:
    return ModuleContext(
        runtime=ModuleRuntimeView(config=object()),
        manifest=ModuleManifest(
            schema_version=1,
            module_id="test-module:v0",
            name="test module",
            version="v0",
            entrypoint="test_module:register_module",
            provides=ModuleProvides(tools=declared_tools),
            sha256="0" * 64,
        ),
    )


def test_module_context_tool_registration_and_summary_generate_specs_linearly() -> None:
    names = tuple(f"tool_{index:03d}" for index in range(100))
    calls: Counter[str] = Counter()
    context = _context(declared_tools=names)

    for name in names:
        context.register_tool(_CountingTool(name, calls))  # type: ignore[arg-type]

    assert context.registered_tool_names == names
    assert context.registered_summary()["tools"] == list(names)
    assert calls == Counter({name: 1 for name in names})


def test_module_context_duplicate_tool_check_does_not_rescan_existing_specs() -> None:
    names = ("first", "second")
    calls: Counter[str] = Counter()
    context = _context(declared_tools=names)
    context.register_tool(_CountingTool("first", calls))  # type: ignore[arg-type]
    context.register_tool(_CountingTool("second", calls))  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="module registered duplicate tool: first"):
        context.register_tool(_CountingTool("first", calls))  # type: ignore[arg-type]

    assert calls == Counter({"first": 2, "second": 1})
    assert context.registered_tool_names == names


def test_module_context_rejects_direct_tool_buffer_mutation() -> None:
    calls: Counter[str] = Counter()
    context = _context(declared_tools=("registered", "bypassed"))
    context.register_tool(_CountingTool("registered", calls))  # type: ignore[arg-type]

    context.tools.append(_CountingTool("bypassed", calls))  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="added through register_tool"):
        context.registered_summary()


@pytest.mark.parametrize(
    ("surface", "expected_registration_method"),
    [
        ("images", "register_image"),
        ("syscalls", "register_syscall"),
        ("provider_hooks", "register_provider_hook"),
        ("startup_hooks", "add_startup_hook"),
        (
            "durable_object_release_finalizers",
            "bind_durable_object_release_finalizer",
        ),
    ],
)
def test_module_context_rejects_direct_registration_buffer_mutation(
    surface: str,
    expected_registration_method: str,
) -> None:
    context = _context(declared_tools=())
    if surface == "images":
        context.images.append(AgentImage(image_id="hidden:v0", name="hidden"))
    elif surface == "syscalls":
        context.syscalls["hidden.call"] = lambda _session, _args: None
    elif surface == "provider_hooks":
        context.provider_hooks["hidden_provider"].append(lambda _host: None)
    elif surface == "startup_hooks":
        context.startup_hooks["hidden_startup"] = lambda _host: None
    else:
        context.durable_object_release_finalizers["hidden.finalizer"] = (
            lambda *_args: {},
            lambda *_args: None,
        )

    with pytest.raises(ValidationError, match=expected_registration_method):
        context.registered_summary()


def test_runtime_persists_tool_specs_with_the_current_runtime_config() -> None:
    config = replace(
        DEFAULT_CONFIG,
        tools=replace(DEFAULT_CONFIG.tools, standard_timeout_s=17.0),
    )
    runtime = Runtime.open("local", config=config)
    try:
        row = next(
            item
            for item in runtime.store.list_tools()
            if item["name"] == "get_current_time"
        )
        spec = json.loads(row["spec_json"])

        assert spec["policy"]["timeout_s"] == 17.0
    finally:
        runtime.close()
