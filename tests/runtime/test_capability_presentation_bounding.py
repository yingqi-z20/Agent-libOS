from __future__ import annotations

from dataclasses import replace

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import Capability, CapabilityRight
from agent_libos.runtime.runtime import Runtime


def _capability(
    cap_id: str,
    *,
    expires_at: str | None = None,
    parent_cap_id: str | None = None,
) -> Capability:
    return Capability(
        cap_id=cap_id,
        subject="bounded-subject",
        resource=f"object:{cap_id}",
        rights={CapabilityRight.READ.value},
        constraints={},
        issued_by="test",
        issued_at="2026-01-01T00:00:00Z",
        expires_at=expires_at,
        parent_cap_id=parent_cap_id,
    )


def test_active_capability_presentation_pages_until_exact_limit_without_full_decode() -> None:
    config = replace(
        DEFAULT_CONFIG,
        capability=replace(DEFAULT_CONFIG.capability, list_limit=2),
    )
    runtime = Runtime.open("local", config=config)
    try:
        runtime.store.insert_capability(
            _capability("cap-001", expires_at="2020-01-01T00:00:00Z")
        )
        runtime.store.insert_capability(
            _capability("cap-002", parent_cap_id="cap-missing")
        )
        runtime.store.insert_capability(_capability("cap-003"))
        runtime.store.insert_capability(_capability("cap-004"))
        runtime.store.insert_capability(_capability("cap-005"))

        decoded: list[str] = []
        decode = runtime.store._row_to_capability
        runtime.store._row_to_capability = lambda row: (
            decoded.append(str(row["cap_id"])),
            decode(row),
        )[1]

        selected = runtime.capability.list_for_presentation(
            include_inactive=False,
            limit=2,
        )

        assert [capability.cap_id for capability in selected] == [
            "cap-003",
            "cap-004",
        ]
        assert decoded == ["cap-001", "cap-002", "cap-003", "cap-004"]
    finally:
        runtime.close()
