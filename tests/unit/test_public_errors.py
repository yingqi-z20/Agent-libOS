from __future__ import annotations

import asyncio
import hashlib

import pytest

from agent_libos.models.exceptions import (
    CapabilityDenied,
    ProviderHostError,
    ValidationError,
)
from agent_libos.utils.public_errors import (
    internal_exception_observation,
    public_error_envelope,
    public_error_envelope_for_type,
    public_exception_message,
    provider_error_envelope,
)
from agent_libos.utils.serde import dumps


def test_internal_public_error_envelope_is_stable_and_text_free() -> None:
    secret = "PUBLIC_ERROR_TOKEN_secret"
    error = RuntimeError(
        "driver failed at /Users/private/runtime.db with "
        f"dsn=postgresql://agent:{secret}@localhost/runtime; "
        "SQL=SELECT * FROM credentials"
    )

    first = public_error_envelope(error)
    second = public_error_envelope(error)

    assert first == second
    assert first["code"] == "internal_error"
    assert first["error_type"] == "RuntimeError"
    assert first["correlation_id"].startswith("corr_")
    assert public_exception_message(error) == first["message"]
    assert secret not in dumps(first)
    assert "/Users/private/runtime.db" not in dumps(first)
    assert "SELECT * FROM credentials" not in dumps(first)


def test_internal_exception_observation_fingerprints_text_without_a_preview() -> None:
    secret = "Q7v9Mx2Lp8Zc4Nk6Hj3Ds5Wa"
    path = "/Users/private/runtime.db"
    sql = "SELECT * FROM credentials"
    text = f"driver failed at {path}; opaque={secret}; SQL={sql}"
    error = RuntimeError(text)

    observation = internal_exception_observation(error)
    encoded = dumps(observation)

    assert observation == {
        "error_type": "RuntimeError",
        "exception_text": {
            "bytes": len(text.encode("utf-8")),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        },
    }
    assert secret not in encoded
    assert path not in encoded
    assert sql not in encoded


def test_internal_exception_observation_handles_broken_exception_text() -> None:
    class BrokenStringError(RuntimeError):
        def __str__(self) -> str:
            raise RuntimeError("Qp4Lx8Vn2Cm7Rw5K")

    observation = internal_exception_observation(BrokenStringError())

    assert observation["error_type"] == "BrokenStringError"
    assert observation["exception_text"]["bytes"] > 0
    assert len(observation["exception_text"]["sha256"]) == 64
    assert "Qp4Lx8Vn2Cm7Rw5K" not in dumps(observation)


def test_internal_exception_observation_bounds_oversized_diagnostic_work() -> None:
    text = "\N{SNOWMAN}" * 100_000

    observation = internal_exception_observation(RuntimeError(text))

    retained = text[:65_536].encode("utf-8")
    assert observation["exception_text"] == {
        "bytes": len(retained),
        "sha256": hashlib.sha256(retained).hexdigest(),
    }
    assert observation["diagnostic_truncated"] is True


@pytest.mark.parametrize(
    "control_flow",
    [
        KeyboardInterrupt(),
        SystemExit(7),
        asyncio.CancelledError(),
        BaseExceptionGroup(
            "interrupted",
            [KeyboardInterrupt(), RuntimeError("ordinary leaf")],
        ),
    ],
)
def test_public_error_cache_never_suppresses_control_flow(
    control_flow: BaseException,
) -> None:
    class InterruptingCacheError(RuntimeError):
        @property
        def _agent_libos_public_error_envelope(self) -> object:
            raise control_flow

        @_agent_libos_public_error_envelope.setter
        def _agent_libos_public_error_envelope(self, _value: object) -> None:
            raise control_flow

    with pytest.raises(type(control_flow)) as raised:
        public_error_envelope(InterruptingCacheError("failure"))

    assert raised.value is control_flow
    assert (
        public_error_envelope(RuntimeError("after interruption"))["code"]
        == "internal_error"
    )


def test_internal_exception_observation_preserves_interruption_from_args() -> None:
    control_flow = BaseExceptionGroup(
        "interrupted diagnostic",
        [SystemExit(9), RuntimeError("ordinary leaf")],
    )

    class InterruptingArgumentsError(RuntimeError):
        @property
        def args(self) -> tuple[object, ...]:
            raise control_flow

    with pytest.raises(BaseExceptionGroup) as raised:
        internal_exception_observation(InterruptingArgumentsError("failure"))

    assert raised.value is control_flow


def test_internal_exception_observation_does_not_execute_custom_formatter() -> None:
    invoked = False

    class ExpensiveStringError(RuntimeError):
        def __str__(self) -> str:
            nonlocal invoked
            invoked = True
            return "x" * 1_000_000

    observation = internal_exception_observation(ExpensiveStringError("secret"))

    assert invoked is False
    assert observation["exception_text"]["bytes"] == len(
        "exception text is unavailable".encode("utf-8")
    )


def test_public_exception_message_does_not_trust_domain_error_text() -> None:
    secret = "DOMAIN_ERROR_SECRET_V9p4Lm2Q"
    for error in (
        ValidationError(f"validation failed for {secret}"),
        CapabilityDenied(f"authority denied for {secret}"),
    ):
        message = public_exception_message(error)

        assert message.startswith(f"internal_error: {type(error).__name__}")
        assert secret not in message


def test_public_exception_message_does_not_trust_arbitrary_value_errors() -> None:
    secret = "V4n8Qm2Lc7Rp5Xw9"

    message = public_exception_message(ValueError(f"driver diagnostic {secret}"))

    assert message.startswith("internal_error: ValueError")
    assert secret not in message


def test_public_error_envelope_for_type_rejects_untrusted_identifiers() -> None:
    secret = "TYPE_ENVELOPE_SECRET_Q8m2"

    public = public_error_envelope_for_type(
        f"Bad\N{RIGHT-TO-LEFT OVERRIDE}Type{secret}",
        code=f"bad/code/{secret}",
    )

    assert public["code"] == "internal_error"
    assert public["error_type"] == "InternalError"
    assert public["correlation_id"].startswith("corr_")
    assert secret not in dumps(public)


def test_provider_not_started_envelope_reuses_one_correlation_id() -> None:
    provider_error_type = type("ProviderEffectNotStarted", (RuntimeError,), {})
    error = provider_error_type("provider-authored secret")

    first = provider_error_envelope(error)
    message = public_exception_message(error)
    second = provider_error_envelope(error)

    assert first is not None
    assert first == second
    assert message == first["message"]
    assert "provider-authored secret" not in message


@pytest.mark.parametrize(
    "invalid_identifier",
    [
        "bad\nidentifier",
        "/Users/private/provider",
        "bad\u202eidentifier",
        "Err\u043er",
        "x" * 257,
    ],
    ids=("newline", "path", "rtl", "confusable", "too-long"),
)
def test_invalid_provider_host_identifiers_degrade_without_leaking_text(
    invalid_identifier: str,
) -> None:
    secret = "PROVIDER_IDENTIFIER_SECRET_X7q2"
    for field in ("code", "error_type", "correlation_id"):
        values = {
            "code": "provider_error",
            "error_type": "ProviderFailure",
            "correlation_id": "corr_host_minted",
        }
        values[field] = f"{invalid_identifier}{secret}"
        error = ProviderHostError(**values)

        assert provider_error_envelope(error) is None
        public = public_error_envelope(error)

        assert public["code"] == "internal_error"
        assert public["error_type"] == "InternalError"
        assert public["correlation_id"].startswith("corr_")
        assert invalid_identifier not in dumps(public)
        assert secret not in dumps(public)


def test_provider_host_error_subclass_cannot_claim_host_public_identity() -> None:
    class ExtensionClaimedProviderError(ProviderHostError):
        pass

    error = ExtensionClaimedProviderError(
        code="provider_error",
        error_type="ProviderFailure",
        correlation_id="corr_extension_claimed",
    )

    assert provider_error_envelope(error) is None
    public = public_error_envelope(error)

    assert public["code"] == "internal_error"
    assert public["error_type"] == "InternalError"
    assert public["correlation_id"] != "corr_extension_claimed"
