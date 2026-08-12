"""Shared fail-closed policy for MCP Apps-only selectors and metadata."""

from __future__ import annotations

from agent_libos.models.exceptions import ValidationError


_APP_META_PREFIX = "io.modelcontextprotocol/ui"


def is_mcp_app_mime(value: str | None) -> bool:
    """Recognize the Apps HTML profile despite case, quotes, and whitespace."""

    if type(value) is not str:
        return False
    parts = [part.strip() for part in value.split(";")]
    if not parts or parts[0].casefold() != "text/html":
        return False
    for parameter in parts[1:]:
        name, separator, selected = parameter.partition("=")
        if (
            separator
            and name.strip().casefold() == "profile"
            and selected.strip().strip('"\'').casefold() == "mcp-app"
        ):
            return True
    return False


def is_mcp_app_metadata_key(value: str) -> bool:
    if type(value) is not str:
        return False
    folded = value.casefold()
    return (
        folded == "ui"
        or folded.startswith("ui/")
        or folded.startswith(_APP_META_PREFIX)
    )


def reject_mcp_app_selector(value: str, *, label: str = "resource selector") -> None:
    if type(value) is not str or not value or "\x00" in value:
        raise ValidationError(f"MCP {label} must be a non-empty string")
    if value.casefold().startswith("ui:"):
        raise ValidationError("MCP Apps ui:// resources are unsupported")


def reject_mcp_app_text(value: str) -> None:
    """Reject an Apps selector/MIME wherever untrusted JSON embeds it."""

    if type(value) is not str:
        raise TypeError("MCP Apps text policy requires a string")
    if value and value.casefold().startswith("ui:"):
        reject_mcp_app_selector(value, label="Apps selector")
    if is_mcp_app_mime(value):
        raise ValidationError("MCP Apps HTML content is unsupported")


__all__ = [
    "is_mcp_app_metadata_key",
    "is_mcp_app_mime",
    "reject_mcp_app_selector",
    "reject_mcp_app_text",
]
