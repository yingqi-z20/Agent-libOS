"""Compatibility exports for the original default-agent module path."""

from __future__ import annotations

from agent_libos.images.default_agents import (
    ANALYSIS_AGENT_PROMPT,
    BASE_AGENT_PROMPT,
    CODING_AGENT_PROMPT,
    CONTEXT_COMPRESSOR_PROMPT,
    DEFAULT_IMAGES,
    MAINTENANCE_AGENT_PROMPT,
    OPERATOR_AGENT_PROMPT,
    RESEARCH_AGENT_PROMPT,
    REVIEW_AGENT_PROMPT,
    TOOLMAKER_AGENT_PROMPT,
    build_default_images,
)

__all__ = [
    "ANALYSIS_AGENT_PROMPT",
    "BASE_AGENT_PROMPT",
    "CODING_AGENT_PROMPT",
    "CONTEXT_COMPRESSOR_PROMPT",
    "DEFAULT_IMAGES",
    "MAINTENANCE_AGENT_PROMPT",
    "OPERATOR_AGENT_PROMPT",
    "RESEARCH_AGENT_PROMPT",
    "REVIEW_AGENT_PROMPT",
    "TOOLMAKER_AGENT_PROMPT",
    "build_default_images",
]
