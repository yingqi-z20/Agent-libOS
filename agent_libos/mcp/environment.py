"""Secret-safe value object for primitive-owned MCP environment snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from agent_libos.models.exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class McpTransportEnvironmentSnapshot:
    """One immutable operation input snapshot; values never enter public data."""

    runtime_environment: Mapping[str, str] = field(repr=False)
    sensitive_values: tuple[str, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_environment, Mapping) or any(
            type(key) is not str
            or not key
            or type(value) is not str
            or "\x00" in value
            for key, value in self.runtime_environment.items()
        ):
            raise ValidationError("MCP runtime environment snapshot is invalid")
        if type(self.sensitive_values) is not tuple or any(
            type(value) is not str or not value for value in self.sensitive_values
        ):
            raise ValidationError("MCP sensitive-value snapshot is invalid")
        object.__setattr__(
            self,
            "runtime_environment",
            MappingProxyType(dict(self.runtime_environment)),
        )
        object.__setattr__(
            self,
            "sensitive_values",
            tuple(dict.fromkeys(self.sensitive_values)),
        )


__all__ = ["McpTransportEnvironmentSnapshot"]
