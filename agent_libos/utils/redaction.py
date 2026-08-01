from __future__ import annotations

import re
from collections.abc import Iterable


_SCALAR_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(
        r"(?i)['\"]?\b(api[_-]?key|authorization|auth[_-]?token|password|passwd|secret|session[_-]?token|token)\b['\"]?"
        r"\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;\"'}]+)"
    ),
    re.compile(r"\bsk_(?:live|test)_[A-Za-z0-9_=-]+"),
    re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox(?:b|p|a|r|s)-[A-Za-z0-9-]{12,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(
        r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"
    ),
    re.compile(r"\bSECRET_[A-Za-z0-9_]+\b"),
    re.compile(r"(?i)\b(?:secret|token)[_-][A-Za-z0-9._-]+\b"),
)

_URI_USERINFO_PATTERN = re.compile(
    r"(?i)(?P<scheme>\b[a-z][a-z0-9+.-]*://)(?P<userinfo>[^\s/@]+)@"
)

_HTTP_COOKIE_HEADER_PATTERN = re.compile(
    r"(?im)(?P<header>\b(?:set-cookie|cookie)\s*:)\s*[^\r\n]*"
)


def redact_sensitive_text(
    value: str,
    *,
    sensitive_values: Iterable[str] = (),
) -> str:
    """Return a stable, public-safe projection of provider-controlled text.

    Pattern redaction protects common credential formats. ``sensitive_values``
    additionally binds redaction to the exact Host-resolved secret snapshot
    used for an operation, covering opaque provider keys that have no
    recognizable prefix. Longest-first replacement prevents a prefix value
    from exposing the remainder of a longer credential.
    """

    selected = value
    exact_values = sorted(
        {item for item in sensitive_values if type(item) is str and item},
        key=len,
        reverse=True,
    )
    for secret in exact_values:
        selected = selected.replace(secret, "[redacted]")
    selected = _URI_USERINFO_PATTERN.sub(
        lambda match: f'{match.group("scheme")}[redacted]@',
        selected,
    )
    selected = _HTTP_COOKIE_HEADER_PATTERN.sub(
        lambda match: f'{match.group("header")} [redacted]',
        selected,
    )
    for pattern in _SCALAR_SECRET_PATTERNS:
        selected = pattern.sub("[redacted]", selected)
    return selected
