from __future__ import annotations

import re
from collections.abc import Iterable


_PUBLIC_ERROR_PATTERN = re.compile(
    r"^(?P<code>[A-Za-z0-9._:-]+): "
    r"(?P<error_type>[A-Za-z0-9._:-]+) "
    r"\(correlation_id=(?P<correlation_id>corr_[A-Za-z0-9._:-]+)\)$"
)


def assert_public_error_message(
    value: object,
    *,
    code: str,
    error_type: str,
    forbidden: Iterable[str] = (),
) -> str:
    """Assert the stable model-facing error envelope and return its correlation id."""

    assert isinstance(value, str)
    matched = _PUBLIC_ERROR_PATTERN.fullmatch(value)
    assert matched is not None, value
    assert matched.group("code") == code
    assert matched.group("error_type") == error_type
    for text in forbidden:
        assert text not in value
    return matched.group("correlation_id")


__all__ = ["assert_public_error_message"]
