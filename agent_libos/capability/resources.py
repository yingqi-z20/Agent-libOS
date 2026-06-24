from __future__ import annotations

from agent_libos.models import ResourcePattern, ResourceScope
from agent_libos.models.exceptions import CapabilityDenied


class ResourceAuthority:
    """Canonical typed resource parser and matcher.

    Capability resources are not raw prefixes. The parser accepts only typed
    resources and terminal wildcards so a grant for one namespace cannot match
    an adjacent namespace by accident.
    """

    def canonical(self, resource: str) -> str:
        raw = str(resource).strip()
        while "//" in raw:
            raw = raw.replace("//", "/")
        return raw.rstrip("/") if raw.endswith("/") and not raw.endswith(":/") else raw

    def parse(self, resource: str, *, requested: bool = False) -> ResourcePattern:
        raw = self.canonical(resource)
        if raw == "*":
            raise CapabilityDenied("capability resources must be typed; use '<kind>:*' instead of global '*'")
        if "*" in raw and not (raw.endswith(":*") or raw.endswith("/*")):
            raise CapabilityDenied(f"resource wildcard must be terminal and canonical: {resource}")
        if ":" not in raw:
            raise CapabilityDenied(f"resource must be typed: {resource}")
        kind, body = raw.split(":", 1)
        if not kind or not body:
            raise CapabilityDenied(f"resource must be typed: {resource}")
        if raw.endswith(":*"):
            if requested and raw != resource:
                raise CapabilityDenied(f"requested resource is not canonical: {resource}")
            return ResourcePattern(raw=raw, kind=kind, body=body[:-2], scope=ResourceScope.PREFIX)
        if raw.endswith("/*"):
            return ResourcePattern(raw=raw, kind=kind, body=body[:-2].rstrip("/"), scope=ResourceScope.SUBTREE)
        return ResourcePattern(raw=raw, kind=kind, body=body, scope=ResourceScope.EXACT)

    def matches(self, granted: str, requested: str) -> bool:
        try:
            pattern = self.parse(granted)
            request = self.parse(requested, requested=True)
        except CapabilityDenied:
            return False
        if pattern.scope == ResourceScope.GLOBAL:
            return True
        if pattern.kind != request.kind:
            return False
        if pattern.scope == ResourceScope.EXACT:
            return pattern.raw == request.raw
        if pattern.scope == ResourceScope.PREFIX:
            if not pattern.body:
                return request.raw.startswith(f"{pattern.kind}:")
            return request.raw.startswith(f"{pattern.kind}:{pattern.body}:") or request.raw == f"{pattern.kind}:{pattern.body}"
        if pattern.scope == ResourceScope.SUBTREE:
            body = request.body
            return body == pattern.body or body.startswith(f"{pattern.body}/")
        return False

    def covers(self, granted: str, requested_pattern: str) -> bool:
        if granted == requested_pattern:
            return True
        try:
            requested = self.parse(requested_pattern)
        except CapabilityDenied:
            return False
        if requested.scope in {ResourceScope.PREFIX, ResourceScope.SUBTREE}:
            return self.matches(granted, requested.raw)
        return self.matches(granted, requested_pattern)
