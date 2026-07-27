from __future__ import annotations

import pytest

from agent_libos.models.exceptions import ValidationError
from agent_libos.runtime.audit_manager import AuditManager
from agent_libos.storage import SQLiteStore


class TestAuditManager:
    @pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.5, "2", 1_001])
    def test_trace_rejects_invalid_limit_before_store_query(
        self,
        invalid: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = SQLiteStore(":memory:")
        audit = AuditManager(store)
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(
            store,
            "list_audit",
            lambda **kwargs: calls.append(kwargs),
        )

        with pytest.raises(ValidationError, match="audit trace limit"):
            audit.trace(limit=invalid)  # type: ignore[arg-type]

        assert calls == []

    @pytest.mark.parametrize("field", ["match_any", "include_gui_presentation"])
    @pytest.mark.parametrize("invalid", [0, 1, "false", None])
    def test_trace_rejects_invalid_boolean_before_store_query(
        self,
        field: str,
        invalid: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = SQLiteStore(":memory:")
        audit = AuditManager(store)
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(
            store,
            "list_audit",
            lambda **kwargs: calls.append(kwargs),
        )
        kwargs = {field: invalid}

        with pytest.raises(ValidationError, match=field):
            audit.trace(**kwargs)  # type: ignore[arg-type]

        assert calls == []

    @pytest.mark.parametrize(("field", "invalid"), [("actor", 1), ("target", 1)])
    def test_trace_rejects_invalid_filter_types_before_store_query(
        self,
        field: str,
        invalid: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = SQLiteStore(":memory:")
        audit = AuditManager(store)
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(
            store,
            "list_audit",
            lambda **kwargs: calls.append(kwargs),
        )

        with pytest.raises(ValidationError, match=field):
            audit.trace(**{field: invalid})  # type: ignore[arg-type]

        assert calls == []

    def test_trace_limit_returns_latest_records_in_chronological_order(self) -> None:
        store = SQLiteStore(":memory:")
        audit = AuditManager(store)

        for index in range(5):
            audit.record(actor="pid_test", action=f"audit.{index}", target="process:pid_test")

        records = audit.trace(limit=2)

        assert [record.action for record in records] == ["audit.3", "audit.4"]

    def test_trace_filters_before_applying_limit(self) -> None:
        store = SQLiteStore(":memory:")
        audit = AuditManager(store)
        audit.record(actor="pid_target", action="target.first", target="process:pid_target")
        for index in range(5):
            audit.record(actor="pid_noise", action=f"noise.{index}", target="process:pid_noise")

        records = audit.trace(limit=1, actor="pid_target", target="process:pid_target", match_any=True)

        assert [record.action for record in records] == ["target.first"]
