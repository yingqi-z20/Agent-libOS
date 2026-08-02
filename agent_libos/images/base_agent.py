"""Compatibility exports for the original default-agent module path."""

from __future__ import annotations

from agent_libos.images.default_agents import (
    BASE_AGENT_PROMPT,
    CODING_AGENT_PROMPT,
    CONTEXT_COMPRESSOR_PROMPT,
    DEFAULT_IMAGES,
    REVIEW_AGENT_PROMPT,
    TOOLMAKER_AGENT_PROMPT,
    build_default_images,
)

__all__ = [
    "BASE_AGENT_PROMPT",
    "CODING_AGENT_PROMPT",
    "CONTEXT_COMPRESSOR_PROMPT",
    "DEFAULT_IMAGES",
    "REVIEW_AGENT_PROMPT",
    "TOOLMAKER_AGENT_PROMPT",
    "build_default_images",
]
