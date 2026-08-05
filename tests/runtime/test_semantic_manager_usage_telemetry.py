from __future__ import annotations

import json
import threading
import time

import pytest

from agent_libos.config import SemanticDefaults
from agent_libos.models import DataLabels
from agent_libos.models.semantic import (
    AuthoritativeApprovalFacts,
    SemanticAssessment,
    SemanticAssessmentKind,
    SemanticAssessmentRequest,
    SemanticAssessmentStatus,
    SemanticDomain,
)
from agent_libos.semantic.external import (
    SemanticProviderResponseError,
    SemanticUsageTelemetry,
)
from agent_libos.semantic.service import SemanticManager
from agent_libos.storage import SQLiteStore, UnitOfWork


pytestmark = pytest.mark.runtime

_DIGEST = "1" * 64
_SECRET = "SEMANTIC_USAGE_STORAGE_SECRET_SENTINEL"


class _TelemetryAssessor:
    def __init__(
        self,
        values: dict[str, SemanticUsageTelemetry],
        *,
        invalid_pids: frozenset[str] = frozenset(),
        barrier: threading.Barrier | None = None,
    ) -> None:
        self._values = values
        self._invalid_pids = invalid_pids
        self._barrier = barrier
        self._local = threading.local()

    def assess(self, request: SemanticAssessmentRequest) -> SemanticAssessment:
        assert request.pid is not None
        self._local.value = self._values[request.pid]
        if self._barrier is not None:
            self._barrier.wait(timeout=5)
        if request.pid in self._invalid_pids:
            raise SemanticProviderResponseError("provider response was invalid")
        return SemanticAssessment(status=SemanticAssessmentStatus.SUCCESS)

    def take_last_usage_telemetry(self) -> SemanticUsageTelemetry | None:
        value = getattr(self._local, "value", None)
        if hasattr(self._local, "value"):
            del self._local.value
        return value


def _request(pid: str) -> SemanticAssessmentRequest:
    return SemanticAssessmentRequest(
        kind=SemanticAssessmentKind.ROOT_GOAL,
        domain=SemanticDomain.RUNTIME,
        action_id="runtime.root_goal",
        input_sha256=_DIGEST,
        deadline_at="2099-01-01T00:00:00+00:00",
        data_labels=DataLabels(),
        features=AuthoritativeApprovalFacts(schema_valid=True),
        pid=pid,
        policy_sha256=_DIGEST,
    )


def _manager(
    unit: UnitOfWork,
    assessor: object,
    *,
    max_concurrency: int = 2,
    adapter: str = "external",
) -> SemanticManager:
    return SemanticManager(
        unit.semantic,
        config=SemanticDefaults(
            mode="shadow",
            adapter=adapter,
            external_profile_id="classifier" if adapter == "external" else None,
            max_concurrency=max_concurrency,
        ),
        assessor=assessor,
        request_capture_registrar=lambda _callback: None,
        spawn_observer_registrar=lambda _callback: None,
        result_observer_registrar=lambda _callback: None,
        request_capture=lambda _request: None,
        spawn_observer=lambda *_args, **_kwargs: None,
        result_observer=lambda *_args, **_kwargs: None,
    )


@pytest.mark.parametrize(
    ("invalid_schema", "expected_status"),
    ((False, "success"), (True, "invalid_schema")),
)
def test_worker_persists_usage_after_any_completed_provider_response(
    invalid_schema: bool,
    expected_status: str,
) -> None:
    pid = "pid-invalid" if invalid_schema else "pid-success"
    expected = SemanticUsageTelemetry(31, 4, 73)
    assessor = _TelemetryAssessor(
        {pid: expected},
        invalid_pids=frozenset({pid}) if invalid_schema else frozenset(),
    )
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        manager = _manager(unit, assessor, max_concurrency=1)
        job = manager._enqueue(  # noqa: SLF001 - exercise terminal producer
            _request(pid),
            candidate=None,
            hard_violations=(),
        )

        assert manager.process_one()
        page = unit.semantic.query_semantic_assessments(
            after=None,
            limit=10,
            pid=pid,
        )
        assert len(page.records) == 1
        record = page.records[0]
        assert record.job_id == job.job_id
        assert record.status == expected_status
        assert (
            record.input_tokens,
            record.output_tokens,
            record.cost_microunits,
        ) == (31, 4, 73)
        assert manager.get_assessment(record.assessment_id)["input_tokens"] == 31
        assert assessor.take_last_usage_telemetry() is None
    finally:
        store.close()


def test_worker_keeps_missing_usage_null_for_deterministic_assessor() -> None:
    class DeterministicAssessor:
        def assess(self, _request: SemanticAssessmentRequest) -> SemanticAssessment:
            return SemanticAssessment(status=SemanticAssessmentStatus.SUCCESS)

    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        manager = _manager(unit, DeterministicAssessor(), max_concurrency=1)
        manager._enqueue(  # noqa: SLF001 - exercise terminal producer
            _request("pid-no-usage"),
            candidate=None,
            hard_violations=(),
        )

        assert manager.process_one()
        record = unit.semantic.query_semantic_assessments(
            after=None,
            limit=10,
            pid="pid-no-usage",
        ).records[0]
        assert record.input_tokens is None
        assert record.output_tokens is None
        assert record.cost_microunits is None
    finally:
        store.close()


def test_scripted_assessor_usage_accessor_is_never_touched() -> None:
    class ScriptedAssessor:
        @property
        def take_last_usage_telemetry(self) -> object:
            raise AssertionError("scripted assessor usage accessor must not be read")

        def assess(self, _request: SemanticAssessmentRequest) -> SemanticAssessment:
            return SemanticAssessment(status=SemanticAssessmentStatus.SUCCESS)

    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        manager = _manager(
            unit,
            ScriptedAssessor(),
            max_concurrency=1,
            adapter="scripted",
        )
        manager._enqueue(  # noqa: SLF001 - exercise adapter isolation
            _request("pid-scripted-no-usage"),
            candidate=None,
            hard_violations=(),
        )

        assert manager.process_one()
        record = unit.semantic.query_semantic_assessments(
            after=None,
            limit=10,
            pid="pid-scripted-no-usage",
        ).records[0]
        assert record.status == "success"
        assert record.input_tokens is None
        assert record.output_tokens is None
        assert record.cost_microunits is None
    finally:
        store.close()


def test_worker_rejects_non_typed_usage_without_persisting_provider_fields() -> None:
    class InvalidTelemetryAssessor:
        def assess(self, _request: SemanticAssessmentRequest) -> SemanticAssessment:
            return SemanticAssessment(status=SemanticAssessmentStatus.SUCCESS)

        def take_last_usage_telemetry(self) -> dict[str, object]:
            return {
                "input_tokens": 17,
                "private_provider_field": _SECRET,
            }

    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        manager = _manager(unit, InvalidTelemetryAssessor(), max_concurrency=1)
        manager._enqueue(  # noqa: SLF001 - exercise terminal producer
            _request("pid-invalid-usage"),
            candidate=None,
            hard_violations=(),
        )

        assert manager.process_one()
        record = unit.semantic.query_semantic_assessments(
            after=None,
            limit=10,
            pid="pid-invalid-usage",
        ).records[0]
        assert record.input_tokens is None
        assert record.output_tokens is None
        assert record.cost_microunits is None
        assert _SECRET not in json.dumps(record.to_dict(), sort_keys=True)
        assert _SECRET not in json.dumps(
            manager.get_assessment(record.assessment_id),
            sort_keys=True,
        )
    finally:
        store.close()


def test_two_workers_do_not_cross_assign_usage() -> None:
    expected = {
        "pid-first": SemanticUsageTelemetry(3, 2, 1),
        "pid-second": SemanticUsageTelemetry(29, 11, 7),
    }
    assessor = _TelemetryAssessor(expected, barrier=threading.Barrier(2))
    store = SQLiteStore(":memory:")
    manager: SemanticManager | None = None
    try:
        unit = UnitOfWork(store)
        manager = _manager(unit, assessor, max_concurrency=2)
        for pid in expected:
            manager._enqueue(  # noqa: SLF001 - exercise concurrent workers
                _request(pid),
                candidate=None,
                hard_violations=(),
            )

        manager.start()
        deadline = time.monotonic() + 5
        records = ()
        while time.monotonic() < deadline:
            records = unit.semantic.query_semantic_assessments(
                after=None,
                limit=10,
            ).records
            if len(records) == 2:
                break
            time.sleep(0.01)
        assert len(records) == 2
        observed = {
            record.pid: SemanticUsageTelemetry(
                record.input_tokens,
                record.output_tokens,
                record.cost_microunits,
            )
            for record in records
        }
        assert observed == expected
    finally:
        if manager is not None:
            assert manager.shutdown()
        store.close()
