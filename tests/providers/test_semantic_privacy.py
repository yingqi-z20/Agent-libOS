from __future__ import annotations

import json
import hashlib
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.config import AgentLibOSConfig, SemanticDefaults
from agent_libos.models import (
    CapabilityRight,
    DataLabels,
    ObjectMetadata,
    ObjectType,
    SemanticAssessment,
    SemanticAssessmentStatus,
)
from agent_libos.storage import SQLiteStore, SemanticAssessmentJobStatus
from agent_libos.substrate import LocalResourceProviderSubstrate
from agent_libos.utils.serde import to_jsonable


pytestmark = pytest.mark.providers

_SECRET_SENTINEL = "ghp_SEMANTICPRIVACYSECRET8d9c7483"
_TERMINAL_JOB_STATUSES = tuple(
    status.value
    for status in SemanticAssessmentJobStatus
    if status
    not in {
        SemanticAssessmentJobStatus.QUEUED,
        SemanticAssessmentJobStatus.CLAIMED,
    }
)


class _BlockingRootAssessor:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.requests: list[Any] = []

    def assess(self, request: Any) -> SemanticAssessment:
        if request.kind.value == "root_goal":
            if request.redacted_intent == "semantic privacy seed":
                return SemanticAssessment(status=SemanticAssessmentStatus.SUCCESS)
            self.requests.append(request)
            self.started.set()
            if not self.release.wait(timeout=5):
                raise TimeoutError("test did not release root assessment")
        return SemanticAssessment(status=SemanticAssessmentStatus.SUCCESS)


def _json_text(value: Any) -> str:
    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_json_text(value).encode("utf-8")).hexdigest()


def _semantic_status(runtime: Runtime) -> dict[str, Any]:
    value = to_jsonable(runtime.semantic.status())
    assert isinstance(value, dict)
    return value


def _wait_for_semantic_idle(
    runtime: Runtime,
    *,
    minimum_assessments: int,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        latest = _semantic_status(runtime)
        queue = latest.get("queue")
        assessments = latest.get("assessments")
        if (
            isinstance(queue, dict)
            and isinstance(assessments, dict)
            and queue.get("queued") == 0
            and queue.get("leased") == 0
            and int(assessments.get("total") or 0) >= minimum_assessments
        ):
            return latest
        time.sleep(0.01)
    raise AssertionError(f"semantic worker did not become idle: {latest}")


def _query_all_assessments(runtime: Runtime) -> dict[str, Any]:
    value = to_jsonable(
        runtime.semantic.query_assessments(
            pid=None,
            request_id=None,
            operation_id=None,
            kind=None,
            status=None,
            domain=None,
            after=None,
            limit=100,
        )
    )
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize(
    ("goal", "source_labels", "expected_mode", "expected_intent"),
    (
        (
            "\x01  Review quarterly trends  \x7f",
            None,
            "redacted",
            "Review quarterly trends",
        ),
        (
            "Review /private/company/report.txt",
            None,
            "metadata_only",
            None,
        ),
        ("Review reports/payroll.csv", None, "metadata_only", None),
        ("Inspect src/auth.py", None, "metadata_only", None),
        ("Inspect ../x", None, "metadata_only", None),
        ("Inspect .ssh/id_rsa", None, "metadata_only", None),
        (r"Inspect src\auth.py", None, "metadata_only", None),
        (
            "Use token=semantic-secret-value-123456",
            None,
            "metadata_only",
            None,
        ),
        (
            "-----BEGIN PRIVATE KEY-----\n"
            + ("A" * 2_100)
            + "\n-----END PRIVATE KEY-----",
            None,
            "metadata_only",
            None,
        ),
        (
            "Classify the confidential launch plan",
            DataLabels(sensitivity="secret"),
            "metadata_only",
            None,
        ),
    ),
)
def test_root_goal_intent_projection_is_bounded_and_privacy_gated(
    goal: str,
    source_labels: DataLabels | None,
    expected_mode: str,
    expected_intent: str | None,
) -> None:
    assessor = _BlockingRootAssessor()
    config = AgentLibOSConfig(
        semantic=SemanticDefaults(
            mode="shadow",
            adapter="scripted",
            max_concurrency=1,
            intent_max_chars=64,
        )
    )
    with TemporaryDirectory() as temp_dir:
        runtime = Runtime.open(
            "local",
            config=config,
            substrate=LocalResourceProviderSubstrate(Path(temp_dir)),
            semantic_assessor=assessor,
        )
        try:
            if source_labels is None:
                pid = runtime.process.spawn(goal=goal)
            else:
                seed_pid = runtime.process.spawn(goal="semantic privacy seed")
                _wait_for_semantic_idle(runtime, minimum_assessments=1)
                goal_handle = runtime.memory.create_object(
                    seed_pid,
                    ObjectType.GOAL,
                    {"text": goal},
                    metadata=ObjectMetadata(
                        sensitivity=source_labels.sensitivity,
                        trust_level=source_labels.trust_level,
                        integrity=source_labels.integrity,
                        origin=source_labels.origin,
                    ),
                    immutable=True,
                )
                stored_goal = runtime.uow.objects.get_object(goal_handle.oid)
                assert stored_goal is not None
                pid = seed_pid
                assert runtime.semantic.capture_root_process(
                    pid,
                    goal=stored_goal,
                ) is not None
            assert assessor.started.wait(timeout=5)
            jobs = runtime.uow.semantic.query_semantic_assessment_jobs(
                statuses=(SemanticAssessmentJobStatus.CLAIMED.value,),
                projection_expires_before=None,
                limit=10,
            )
            selected = [job for job in jobs if job.pid == pid]
            assert len(selected) == 1
            projection = selected[0].projection
            assert projection["projection_mode"] == expected_mode
            assert projection.get("redacted_intent") == expected_intent
            assert assessor.requests[0].redacted_intent == expected_intent
            if expected_intent is None:
                assert goal not in _json_text(projection)
        finally:
            assessor.release.set()
            runtime.close()


def test_terminal_semantic_evidence_never_persists_secret_sentinel() -> None:
    config = AgentLibOSConfig(
        semantic=SemanticDefaults(mode="shadow", adapter="deterministic")
    )
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        db_path = root / "runtime.sqlite"
        workspace = root / "workspace"
        workspace.mkdir()
        secret_path = workspace / "provider-secret.txt"
        secret_path.write_text(_SECRET_SENTINEL, encoding="utf-8")
        store = SQLiteStore(db_path)
        runtime = Runtime(
            store,
            config=config,
            substrate=LocalResourceProviderSubstrate(workspace),
        )
        try:
            pid = runtime.process.spawn(
                goal=f"Classify this goal without retaining {_SECRET_SENTINEL}"
            )
            runtime.filesystem.grant_path(
                pid,
                "provider-secret.txt",
                [CapabilityRight.READ],
                issued_by="test.host",
            )
            result = runtime.filesystem.read_text(pid, "provider-secret.txt")
            assert result.content == _SECRET_SENTINEL
            bytes_result = runtime.filesystem.read_bytes(pid, "provider-secret.txt")
            assert bytes_result.content == _SECRET_SENTINEL.encode("utf-8")
            binding_after_result = runtime.store.get_file_label_binding(
                "provider-secret.txt"
            )
            context_after_result = runtime.data_flow.current_context()
            _wait_for_semantic_idle(runtime, minimum_assessments=3)

            jobs = runtime.uow.semantic.query_semantic_assessment_jobs(
                statuses=_TERMINAL_JOB_STATUSES,
                projection_expires_before=None,
                limit=500,
            )
            assert len(jobs) >= 3
            assert all(job.projection == {} for job in jobs)
            assert all(job.projection_retention.value == "hash_only" for job in jobs)
            assessments = runtime.uow.semantic.query_semantic_assessments(
                after=None,
                limit=500,
            ).records
            assert len(assessments) >= 3
            root_data_findings = [
                item
                for assessment in assessments
                if assessment.kind == "root_goal"
                for item in assessment.data_findings
            ]
            provider_data_findings = [
                item
                for assessment in assessments
                if assessment.kind == "provider_ingress"
                for item in assessment.data_findings
            ]
            assert any(
                item["category"] == "credential"
                and item["field"] == "root_goal"
                and item["sensitivity_floor"] == "secret"
                for item in root_data_findings
            )
            assert sum(
                item["category"] == "credential"
                and item["field"] == "provider.result"
                and item["sensitivity_floor"] == "secret"
                and item["integrity_ceiling"] == "untrusted"
                and item["trust_ceiling"] == "untrusted"
                for item in provider_data_findings
            ) >= 2
            assert runtime.store.get_file_label_binding(
                "provider-secret.txt"
            ) == binding_after_result
            assert runtime.data_flow.current_context() == context_after_result

            api_page = _query_all_assessments(runtime)
            api_items = api_page.get("items")
            assert isinstance(api_items, list) and len(api_items) >= 2
            api_details = [
                runtime.semantic.get_assessment(str(item["assessment_id"]))
                for item in api_items
            ]
            raw_semantic_rows = {
                "jobs": [
                    dict(row)
                    for row in runtime.store._query(  # noqa: SLF001 - privacy oracle
                        "SELECT projection_json, bindings_json, error_code "
                        "FROM semantic_assessment_jobs"
                    )
                ],
                "assessments": [
                    dict(row)
                    for row in runtime.store._query(  # noqa: SLF001 - privacy oracle
                        "SELECT record_json FROM semantic_assessments"
                    )
                ],
            }
            surfaces = {
                "typed_jobs": jobs,
                "typed_assessments": assessments,
                "semantic_status": runtime.semantic.status(),
                "semantic_api_page": api_page,
                "semantic_api_details": api_details,
                "semantic_sql_rows": raw_semantic_rows,
                "events": runtime.events.list(),
                "audit": runtime.audit.trace(),
                "llm_calls": runtime.store.list_llm_calls(limit=100),
            }
            assert _SECRET_SENTINEL not in _json_text(surfaces)
        finally:
            runtime.close()


def test_root_goal_and_provider_ingress_are_digest_bound_without_label_writeback() -> None:
    config = AgentLibOSConfig(
        semantic=SemanticDefaults(mode="shadow", adapter="deterministic")
    )
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        goal = "Review the provider result without changing its data labels."
        path = "provider-result.txt"
        content = "committed provider result"
        (root / path).write_text(content, encoding="utf-8")
        runtime = Runtime.open(
            "local",
            config=config,
            substrate=LocalResourceProviderSubstrate(root),
        )
        try:
            pid = runtime.process.spawn(goal=goal)
            runtime.filesystem.grant_path(
                pid,
                path,
                [CapabilityRight.READ],
                issued_by="test.host",
            )
            result = runtime.filesystem.read_text(pid, path)
            binding_after_result = runtime.store.get_file_label_binding(path)
            context_after_result = runtime.data_flow.current_context()
            effects = [
                effect
                for effect in runtime.store.list_external_effects(pid=pid)
                if (
                    effect.provider == "filesystem"
                    and effect.operation == "read_bytes"
                    and (
                        effect.provider_metadata.get("protected_operation") or {}
                    ).get("contract_name")
                    == "primitive.filesystem.read_text"
                )
            ]
            assert len(effects) == 1
            _wait_for_semantic_idle(runtime, minimum_assessments=2)

            records = _semantic_records_for_pid(runtime, pid)
            root_records = [record for record in records if record.kind == "root_goal"]
            provider_records = [
                record
                for record in records
                if record.kind == "provider_ingress"
                and record.effect_id == effects[0].effect_id
            ]
            assert len(root_records) == 1
            assert len(provider_records) == 1
            assert root_records[0].input_sha256 == _canonical_sha256({"text": goal})
            assert provider_records[0].input_sha256 == _canonical_sha256(result)
            assert provider_records[0].input_sha256 != _canonical_sha256(
                effects[0].target
            )
            assert provider_records[0].provider_spec_sha256 is not None
            assert runtime.store.get_file_label_binding(path) == binding_after_result
            assert runtime.data_flow.current_context() == context_after_result
        finally:
            runtime.close()


def _semantic_records_for_pid(runtime: Runtime, pid: str) -> tuple[Any, ...]:
    return runtime.uow.semantic.query_semantic_assessments(
        after=None,
        limit=100,
        pid=pid,
    ).records
