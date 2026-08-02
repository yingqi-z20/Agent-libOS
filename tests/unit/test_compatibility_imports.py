from __future__ import annotations

from agent_libos.images import DEFAULT_IMAGES
from agent_libos.images.base_agent import DEFAULT_IMAGES as COMPAT_DEFAULT_IMAGES
from agent_libos.llm.context_management import estimate_multilingual_tokens
from agent_libos.llm.event_projection import (
    ProjectedEventBatch,
    PromptEventProjection,
)
from agent_libos.llm.openai_schema import openai_chat_tool_schema
from agent_libos.models import RuntimeModule
from agent_libos.models.snapshot import ProcessSnapshot
from agent_libos.modules import LoadedModule
from agent_libos.process_transition import ProcessTransitionService
from agent_libos.runtime.process_transition import (
    ProcessTransitionService as CompatProcessTransitionService,
)
from agent_libos.runtime.snapshots.models import (
    ProcessSnapshot as CompatProcessSnapshot,
)
from agent_libos.utils.ids import estimate_tokens
from agent_libos.utils.openai_schema import (
    openai_chat_tool_schema as current_openai_chat_tool_schema,
)


def test_published_1x_compatibility_imports_remain_identity_preserving() -> None:
    assert COMPAT_DEFAULT_IMAGES is DEFAULT_IMAGES
    assert openai_chat_tool_schema is current_openai_chat_tool_schema
    assert CompatProcessTransitionService is ProcessTransitionService
    assert CompatProcessSnapshot is ProcessSnapshot
    assert LoadedModule is RuntimeModule
    assert PromptEventProjection is ProjectedEventBatch
    assert estimate_multilingual_tokens("你好 abc") == estimate_tokens("你好 abc")


def test_projected_event_compatibility_property_returns_visible_records() -> None:
    visible = [{"type": "example"}]
    batch = ProjectedEventBatch(
        visible_records=visible,
        represented_through_event_id=None,
        omitted_counts={},
        resource_usage_delta={},
        summary={},
    )

    assert batch.events is visible
