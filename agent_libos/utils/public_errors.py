from __future__ import annotations

import hashlib
import threading
from codecs import getincrementalencoder
from dataclasses import dataclass
from typing import Any, Mapping

from agent_libos.models.exceptions import ProviderHostError
from agent_libos.utils.ids import new_id


_PUBLIC_ERROR_CACHE_ATTR = "_agent_libos_public_error_envelope"
_PUBLIC_ERROR_CACHE_MARKER = object()
_PUBLIC_ERROR_CACHE_LOCK = threading.Lock()
_MAX_DIAGNOSTIC_CHARS = 65_536
_UNAVAILABLE_EXCEPTION_TEXT = "exception text is unavailable"


@dataclass(frozen=True, slots=True)
class PublicErrorEnvelope:
    """Model-visible provider failure identity with no provider-authored text."""

    code: str
    error_type: str
    correlation_id: str

    @property
    def message(self) -> str:
        return (
            f"{self.code}: {self.error_type} "
            f"(correlation_id={self.correlation_id})"
        )

    def to_dict(self, *, include_message: bool = False) -> dict[str, str]:
        payload = {
            "code": self.code,
            "error_type": self.error_type,
            "correlation_id": self.correlation_id,
        }
        if include_message:
            payload["message"] = self.message
        return payload

    @classmethod
    def from_error(cls, error: BaseException) -> PublicErrorEnvelope | None:
        if type(error) is ProviderHostError:
            if not all(
                _is_public_identifier(value)
                for value in (
                    error.code,
                    error.error_type,
                    error.correlation_id,
                )
            ):
                return None
            return cls(
                code=error.code,
                error_type=error.error_type,
                correlation_id=error.correlation_id,
            )
        if type(error).__name__ == "ProviderEffectNotStarted":
            return cls(
                code="provider_effect_not_started",
                error_type=type(error).__name__,
                correlation_id=new_id("corr"),
            )
        return None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PublicErrorEnvelope | None:
        selected = {
            key: value.get(key)
            for key in ("code", "error_type", "correlation_id")
        }
        if not all(_is_public_identifier(item) for item in selected.values()):
            return None
        return cls(
            code=str(selected["code"]),
            error_type=str(selected["error_type"]),
            correlation_id=str(selected["correlation_id"]),
        )


def provider_error_envelope(error: BaseException) -> dict[str, str] | None:
    """Return a stable public envelope without provider-authored text."""

    selected = PublicErrorEnvelope.from_error(error)
    if selected is None:
        return None
    with _PUBLIC_ERROR_CACHE_LOCK:
        envelope = _cached_public_error_envelope(error) or selected
        _cache_public_error_envelope(error, envelope)
    return envelope.to_dict(include_message=True) if envelope is not None else None


def provider_error_envelope_from_mapping(
    value: Mapping[str, Any],
) -> dict[str, str] | None:
    """Recover a validated public envelope from an internal protocol frame."""

    envelope = PublicErrorEnvelope.from_mapping(value)
    return envelope.to_dict(include_message=True) if envelope is not None else None


def public_error_envelope(
    error: BaseException,
    *,
    code: str = "internal_error",
) -> dict[str, str]:
    """Return one stable, text-free envelope for an outward-facing failure."""

    with _PUBLIC_ERROR_CACHE_LOCK:
        envelope = _cached_public_error_envelope(error)
        if envelope is None:
            envelope = PublicErrorEnvelope.from_error(error)
        if envelope is None and isinstance(error, ProviderHostError):
            # A ProviderHostError is public only when every Host-minted field
            # satisfies the closed identifier grammar.  Never reuse a partial
            # or malformed provider identity in the generic fallback.  This
            # also rejects subclasses: extension code must not gain a public
            # identity merely by inheriting the Host-owned exception class.
            envelope = PublicErrorEnvelope(
                code="internal_error",
                error_type="InternalError",
                correlation_id=new_id("corr"),
            )
        if envelope is None:
            error_type = type(error).__name__
            envelope = PublicErrorEnvelope(
                code=code if _is_public_identifier(code) else "internal_error",
                error_type=(
                    error_type
                    if _is_public_identifier(error_type)
                    else "InternalError"
                ),
                correlation_id=new_id("corr"),
            )
        _cache_public_error_envelope(error, envelope)
    return envelope.to_dict(include_message=True)


def public_error_envelope_for_type(
    error_type: str,
    *,
    code: str = "internal_error",
) -> dict[str, str]:
    """Mint a text-free envelope without consulting an exception instance.

    This is intended for trust boundaries where even exception attributes may
    be controlled by untrusted extension code.  Both caller-supplied identity
    fields are constrained to the same closed ASCII grammar as recovered
    provider envelopes.
    """

    envelope = PublicErrorEnvelope(
        code=code if _is_public_identifier(code) else "internal_error",
        error_type=(
            error_type
            if _is_public_identifier(error_type)
            else "InternalError"
        ),
        correlation_id=new_id("corr"),
    )
    return envelope.to_dict(include_message=True)


def internal_error_observation(
    *,
    error_type: str,
    text: str,
    correlation_id: str,
) -> dict[str, Any]:
    """Fingerprint private diagnostic text and bind it to a public envelope."""

    selected_type = (
        error_type if _is_public_identifier(error_type) else "InternalError"
    )
    selected_correlation = (
        correlation_id
        if _is_public_identifier(correlation_id)
        else "correlation_unavailable"
    )
    encoded_bytes, encoded_sha256, truncated = _utf8_fingerprint(text)
    observation: dict[str, Any] = {
        "error_type": selected_type,
        "correlation_id": selected_correlation,
        "exception_text": {
            "bytes": encoded_bytes,
            "sha256": encoded_sha256,
        },
    }
    if truncated:
        observation["diagnostic_truncated"] = True
    return observation


def internal_exception_observation(
    error: BaseException,
    *,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """Fingerprint internal text without persisting provider-authored bytes."""

    error_type = type(error).__name__
    observation = internal_error_observation(
        error_type=error_type,
        text=_safe_exception_text(error),
        correlation_id=correlation_id or "correlation_unavailable",
    )
    if correlation_id is None:
        observation.pop("correlation_id", None)
    return observation


def public_exception_message(error: BaseException) -> str:
    return public_error_envelope(error)["message"]


def _cached_public_error_envelope(
    error: BaseException,
) -> PublicErrorEnvelope | None:
    try:
        attributes = object.__getattribute__(error, "__dict__")
    except BaseException as exc:
        _reraise_control_flow(exc)
        return None
    if not isinstance(attributes, dict):
        return None
    cached = attributes.get(_PUBLIC_ERROR_CACHE_ATTR)
    if (
        isinstance(cached, tuple)
        and len(cached) == 2
        and cached[0] is _PUBLIC_ERROR_CACHE_MARKER
        and isinstance(cached[1], PublicErrorEnvelope)
    ):
        return cached[1]
    return None


def _cache_public_error_envelope(
    error: BaseException,
    envelope: PublicErrorEnvelope,
) -> None:
    try:
        object.__setattr__(
            error,
            _PUBLIC_ERROR_CACHE_ATTR,
            (_PUBLIC_ERROR_CACHE_MARKER, envelope),
        )
    except BaseException as exc:
        _reraise_control_flow(exc)
        return


def _safe_exception_text(error: BaseException) -> str:
    """Format ordinary exception arguments without invoking extension code.

    The common ``Exception("message")`` case retains its exact diagnostic.
    Custom ``__str__`` implementations and non-string argument graphs are not
    executed from this ubiquitous failure path because they have no bounded
    time or allocation contract.
    """

    try:
        formatter = type(error).__str__
        arguments = object.__getattribute__(error, "args")
    except BaseException as exc:
        _reraise_control_flow(exc)
        return _UNAVAILABLE_EXCEPTION_TEXT
    if formatter is not BaseException.__str__:
        return _UNAVAILABLE_EXCEPTION_TEXT
    if not isinstance(arguments, tuple):
        return _UNAVAILABLE_EXCEPTION_TEXT
    if not arguments:
        return ""
    if len(arguments) == 1 and type(arguments[0]) is str:
        return arguments[0]
    return _UNAVAILABLE_EXCEPTION_TEXT


def _reraise_control_flow(error: BaseException) -> None:
    """Never turn interpreter/cancellation control flow into diagnostics."""

    if not isinstance(error, Exception):
        raise error


def _is_public_identifier(value: Any) -> bool:
    if not isinstance(value, str) or not value or len(value) > 256:
        return False
    return all(
        "a" <= character <= "z"
        or "A" <= character <= "Z"
        or "0" <= character <= "9"
        or character in "._:-"
        for character in value
    )


def _utf8_fingerprint(value: str) -> tuple[int, str, bool]:
    """Hash a bounded UTF-8 prefix without creating a second huge copy."""

    encoder = getincrementalencoder("utf-8")(errors="replace")
    digest = hashlib.sha256()
    total = 0
    chunk_chars = 8_192
    selected_chars = min(len(value), _MAX_DIAGNOSTIC_CHARS)
    for offset in range(0, selected_chars, chunk_chars):
        encoded = encoder.encode(
            value[offset : min(offset + chunk_chars, selected_chars)],
            final=False,
        )
        total += len(encoded)
        digest.update(encoded)
    tail = encoder.encode("", final=True)
    total += len(tail)
    digest.update(tail)
    return total, digest.hexdigest(), len(value) > selected_chars


__all__ = [
    "PublicErrorEnvelope",
    "internal_error_observation",
    "internal_exception_observation",
    "public_error_envelope",
    "public_error_envelope_for_type",
    "provider_error_envelope",
    "provider_error_envelope_from_mapping",
    "public_exception_message",
]
