"""Host-private local Responses continuation records.

These records are deliberately separate from observable LLM call evidence.
Only local storage and trusted replay assembly should receive their payloads.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any


def canonical_replay_payload(payload: dict[str, Any]) -> str:
    if type(payload) is not dict:
        raise ValueError("LLM replay payload must be a JSON object")
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    # json.dumps silently converts tuples and non-string mapping keys. Reject
    # them so the immutable insert and its subsequent read describe one value.
    pending: list[Any] = [payload]
    while pending:
        item = pending.pop()
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError("LLM replay JSON object keys must be strings")
            pending.extend(item.values())
        elif type(item) is list:
            pending.extend(item)
        elif item is not None and type(item) not in {str, int, float, bool}:
            raise ValueError("LLM replay payload must contain only JSON values")
    return encoded


@dataclass(frozen=True, slots=True)
class LLMReplayTurn:
    turn_id: str
    pid: str
    run_id: str | None
    provider_fingerprint: str
    model: str
    context_generation: str
    payload: dict[str, Any] | None = field(repr=False, metadata={"serialize": False})
    source_labels: dict[str, Any] = field(repr=False, metadata={"serialize": False})
    payload_sha256: str
    payload_bytes: int
    created_at: str
    purged_at: str | None = None

    def __post_init__(self) -> None:
        for name in ("turn_id", "pid", "provider_fingerprint", "model", "context_generation", "created_at"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"LLM replay {name} must be a nonempty string")
        if self.run_id is not None and (not isinstance(self.run_id, str) or not self.run_id):
            raise ValueError("LLM replay run_id must be a nonempty string or None")
        if not isinstance(self.payload_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.payload_sha256):
            raise ValueError("LLM replay payload digest is invalid")
        if type(self.payload_bytes) is not int or self.payload_bytes < 0:
            raise ValueError("LLM replay payload byte count is invalid")
        canonical_replay_payload(self.source_labels)
        if (self.payload is None) != (self.purged_at is not None):
            raise ValueError("LLM replay payload absence must have a purge tombstone")
        if self.payload is not None:
            encoded = canonical_replay_payload(self.payload).encode("utf-8")
            if len(encoded) != self.payload_bytes or hashlib.sha256(encoded).hexdigest() != self.payload_sha256:
                raise ValueError("LLM replay payload integrity mismatch")

    @classmethod
    def from_payload(cls, *, payload: dict[str, Any], **kwargs: Any) -> LLMReplayTurn:
        encoded = canonical_replay_payload(payload).encode("utf-8")
        return cls(payload=payload, payload_sha256=hashlib.sha256(encoded).hexdigest(), payload_bytes=len(encoded), **kwargs)


@dataclass(frozen=True, slots=True)
class LLMReplayHead:
    pid: str
    turn_id: str
    revision: int
    updated_at: str

    def __post_init__(self) -> None:
        for name in ("pid", "turn_id", "updated_at"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"LLM replay head {name} must be a nonempty string")
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("LLM replay head revision must be positive")
