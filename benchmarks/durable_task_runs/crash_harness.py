from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, NoReturn

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import (
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    JsonRpcTransportResult,
    TaskRunStatus,
)
from agent_libos.substrate import LocalResourceProviderSubstrate
from agent_libos.utils.serde import to_jsonable


CRASH_EXIT_CODE = 91
_LEDGER_SCHEMA_VERSION = 1
_MAX_LEDGER_BYTES = 4 * 1024 * 1024
_MAX_RECORD_BYTES = 64 * 1024


class DurabilityBarrier(str, Enum):
    """Crash locations that bracket the Task Run/effect commit protocol."""

    RUN_COMMITTED = "run_committed"
    ACTION_COMMITTED = "action_committed"
    EFFECT_PREPARED = "effect_prepared"
    PROVIDER_DISPATCHED = "provider_dispatched"
    PROVIDER_RESULT_DURABLE = "provider_result_durable"
    RESUME_POINT_COMMITTED = "resume_point_committed"


class RecoveryClass(str, Enum):
    PURE = "pure"
    CERTIFIED_NOT_STARTED = "certified_not_started"
    PROVIDER_IDEMPOTENT = "provider_idempotent"
    UNKNOWN_EFFECT = "unknown_effect"


class ProviderOutcome(str, Enum):
    ABSENT = "absent"
    NOT_STARTED = "not_started"
    SUCCEEDED = "succeeded"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CrashMatrixResult:
    barrier: DurabilityBarrier
    recovery_class: RecoveryClass
    process_returncode: int
    provider_outcome: ProviderOutcome
    dispatch_count: int
    receipt_count: int
    runtime_reopened: bool
    recovered_status: str | None
    blocker_kinds: tuple[str, ...]
    local_effect_transaction_state: str | None
    root_present: bool
    validated_action_present: bool
    tool_call_present: bool
    effect_link_present: bool
    resume_point_present: bool
    pending_action_present: bool
    local_llm_call_count: int
    completed_step_count: int
    settlement_reopen_stable: bool
    idempotency_dedupe_verified: bool
    reopen_evidence_fingerprint: str

    @property
    def passed(self) -> bool:
        if (
            self.process_returncode != _expected_returncode(self.barrier)
            or not self.runtime_reopened
            or not self.root_present
            or not self.settlement_reopen_stable
        ):
            return False
        expected = {
            RecoveryClass.PURE: ProviderOutcome.ABSENT,
            RecoveryClass.CERTIFIED_NOT_STARTED: ProviderOutcome.NOT_STARTED,
            RecoveryClass.PROVIDER_IDEMPOTENT: ProviderOutcome.SUCCEEDED,
        }.get(self.recovery_class)
        if (
            (expected is not None and self.provider_outcome is not expected)
            or self.dispatch_count > 1
        ):
            return False
        if self.recovery_class is RecoveryClass.UNKNOWN_EFFECT:
            common = (
                self.dispatch_count == 1
                and self.recovered_status == TaskRunStatus.NEEDS_ATTENTION.value
                and "unknown_effect" in self.blocker_kinds
            )
            if self.provider_outcome is ProviderOutcome.UNKNOWN:
                return (
                    common
                    and self.receipt_count == 0
                    and self.local_effect_transaction_state
                    in {"dispatched", "unknown"}
                )
            if self.provider_outcome is ProviderOutcome.SUCCEEDED:
                return (
                    common
                    and self.receipt_count == 1
                    and self.local_effect_transaction_state == "committed"
                    and self.resume_point_present
                    and self.pending_action_present
                    and self.local_llm_call_count == 2
                )
            return False
        if self.recovered_status == TaskRunStatus.NEEDS_ATTENTION.value:
            return False
        if self.recovery_class is RecoveryClass.PURE:
            expected_status = (
                TaskRunStatus.QUEUED.value
                if self.barrier is DurabilityBarrier.RUN_COMMITTED
                else TaskRunStatus.PAUSED.value
            )
            return (
                self.recovered_status == expected_status
                and self.dispatch_count == 0
                and self.receipt_count == 0
                and (
                    self.barrier is not DurabilityBarrier.ACTION_COMMITTED
                    or (
                        self.validated_action_present
                        and self.pending_action_present
                        and self.local_llm_call_count == 1
                    )
                )
            )
        if self.recovery_class is RecoveryClass.CERTIFIED_NOT_STARTED:
            return (
                self.recovered_status == TaskRunStatus.PAUSED.value
                and self.dispatch_count == 0
                and self.receipt_count == 0
                and self.local_effect_transaction_state == "failed"
                and self.pending_action_present
            )
        return (
            self.recovered_status == TaskRunStatus.PAUSED.value
            and self.dispatch_count == 1
            and self.receipt_count == 1
            and self.local_effect_transaction_state == "committed"
            and self.tool_call_present
            and self.effect_link_present
            and self.resume_point_present
            and not self.pending_action_present
            and self.local_llm_call_count == 1
            and self.completed_step_count == 1
            and self.idempotency_dedupe_verified
        )


class FsyncProviderLedger:
    """A provider-side ledger whose durability is independent of RuntimeStore.

    Every complete line is one canonical JSON record. A write is acknowledged
    only after the file descriptor is fsynced; initial creation also fsyncs the
    parent directory where the platform permits it. Ledger I/O uses only
    operating-system file descriptors, so a Runtime crash cannot share its
    transactions, buffers, or cleanup with this evidence source.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        if self.path.exists() and not self.path.is_file():
            raise ValueError("provider ledger path must be a regular file")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        *,
        effect_id: str,
        kind: str,
        outcome: ProviderOutcome,
        idempotency_key: str | None = None,
        receipt: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(effect_id, str) or not effect_id:
            raise ValueError("effect_id must be non-empty")
        if not isinstance(kind, str) or not kind:
            raise ValueError("kind must be non-empty")
        existing = self.records()
        if any(
            row["effect_id"] == effect_id and row["kind"] == kind
            for row in existing
        ):
            raise ValueError("provider ledger evidence identity already exists")
        claimed_effects = {
            str(row["effect_id"])
            for row in existing
            if idempotency_key is not None
            and row["idempotency_key"] == idempotency_key
        }
        if claimed_effects and claimed_effects != {effect_id}:
            raise ValueError("provider idempotency key is already claimed")
        record = {
            "schema_version": _LEDGER_SCHEMA_VERSION,
            "sequence": len(existing) + 1,
            "effect_id": effect_id,
            "kind": kind,
            "outcome": outcome.value,
            "idempotency_key": idempotency_key,
            "receipt": receipt,
        }
        encoded = (
            json.dumps(
                record,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        if len(encoded) > _MAX_RECORD_BYTES:
            raise ValueError("provider ledger record exceeds hard limit")
        created = not self.path.exists()
        descriptor = os.open(
            self.path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:  # pragma: no cover - defensive OS contract
                    raise OSError("provider ledger write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if created:
            _fsync_directory(self.path.parent)
        return record

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        if size > _MAX_LEDGER_BYTES:
            raise ValueError("provider ledger exceeds hard limit")
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ValueError("provider ledger has an incomplete record")
        records: list[dict[str, Any]] = []
        for expected_sequence, line in enumerate(raw.splitlines(), start=1):
            if len(line) > _MAX_RECORD_BYTES:
                raise ValueError("provider ledger record exceeds hard limit")
            try:
                decoded = json.loads(
                    line.decode("utf-8"),
                    object_pairs_hook=_unique_json_object,
                    parse_constant=_reject_json_constant,
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError("provider ledger contains invalid JSON") from exc
            canonical = json.dumps(
                decoded,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            if canonical != line:
                raise ValueError("provider ledger record is not canonical JSON")
            if not isinstance(decoded, dict) or set(decoded) != {
                "schema_version",
                "sequence",
                "effect_id",
                "kind",
                "outcome",
                "idempotency_key",
                "receipt",
            }:
                raise ValueError("provider ledger record has invalid shape")
            if (
                type(decoded["schema_version"]) is not int
                or decoded["schema_version"] != _LEDGER_SCHEMA_VERSION
            ):
                raise ValueError("provider ledger record has unsupported schema")
            if (
                type(decoded["sequence"]) is not int
                or decoded["sequence"] != expected_sequence
            ):
                raise ValueError("provider ledger sequence is not contiguous")
            if not isinstance(decoded["effect_id"], str) or not decoded["effect_id"]:
                raise ValueError("provider ledger effect_id is invalid")
            if not isinstance(decoded["kind"], str) or not decoded["kind"]:
                raise ValueError("provider ledger kind is invalid")
            idempotency_key = decoded["idempotency_key"]
            if idempotency_key is not None and (
                not isinstance(idempotency_key, str) or not idempotency_key
            ):
                raise ValueError("provider ledger idempotency_key is invalid")
            if decoded["receipt"] is not None and not isinstance(
                decoded["receipt"], dict
            ):
                raise ValueError("provider ledger receipt is invalid")
            try:
                ProviderOutcome(decoded["outcome"])
            except (TypeError, ValueError) as exc:
                raise ValueError("provider ledger outcome is invalid") from exc
            records.append(decoded)
        identities = [
            (str(row["effect_id"]), str(row["kind"])) for row in records
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("provider ledger contains duplicate effect evidence")
        key_effects: dict[str, set[str]] = {}
        for row in records:
            key = row["idempotency_key"]
            if isinstance(key, str):
                key_effects.setdefault(key, set()).add(str(row["effect_id"]))
        if any(len(effect_ids) != 1 for effect_ids in key_effects.values()):
            raise ValueError("provider ledger idempotency key maps to multiple effects")
        return records

    def classify(self, effect_id: str) -> ProviderOutcome:
        matching = [row for row in self.records() if row["effect_id"] == effect_id]
        if not matching:
            return ProviderOutcome.ABSENT
        return ProviderOutcome(matching[-1]["outcome"])

    def count(self, effect_id: str, kind: str) -> int:
        return sum(
            row["effect_id"] == effect_id and row["kind"] == kind
            for row in self.records()
        )

    def successful_receipt_for_key(
        self,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        matching = [
            row
            for row in self.records()
            if row["idempotency_key"] == idempotency_key
            and row["kind"] == "receipt"
            and row["outcome"] == ProviderOutcome.SUCCEEDED.value
        ]
        if not matching:
            return None
        if len(matching) != 1:
            raise ValueError("provider ledger has ambiguous idempotency receipt")
        receipt = matching[0]["receipt"]
        if not isinstance(receipt, dict):
            raise ValueError("provider ledger success receipt is missing")
        return dict(receipt)


class FsyncIdempotentJsonRpcProvider:
    """JSON-RPC provider with an independent, fsynced idempotency ledger.

    The initial request is reached through the ordinary model-action, tool,
    JSON-RPC primitive, and protected-effect path.  A successor process can
    reconstruct this provider from only the fsynced ledger and prove that the
    same endpoint idempotency key returns the prior receipt without another
    external dispatch.
    """

    def __init__(
        self,
        ledger: FsyncProviderLedger,
        *,
        barrier: DurabilityBarrier | None = None,
        crash: Callable[[bool], None] | None = None,
    ) -> None:
        self._ledger = ledger
        self._barrier = barrier
        self._crash = crash
        self._runtime: Runtime | None = None
        self._pid: str | None = None

    def attach(self, runtime: Runtime, pid: str) -> None:
        self._runtime = runtime
        self._pid = pid

    def call(
        self,
        _endpoint: Any,
        _method: Any,
        request_body: bytes,
        **_kwargs: Any,
    ) -> JsonRpcTransportResult:
        request = json.loads(request_body.decode("utf-8"))
        params = request.get("params")
        if not isinstance(params, dict):
            raise AssertionError("crash provider requires object params")
        idempotency_key = params.get("idempotency_key")
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise AssertionError("crash provider requires an idempotency key")

        prior = self._ledger.successful_receipt_for_key(idempotency_key)
        if prior is not None:
            return self._transport_result(request.get("id"), prior)
        if any(
            row["idempotency_key"] == idempotency_key
            for row in self._ledger.records()
        ):
            raise RuntimeError(
                "provider idempotency outcome is unresolved; redispatch refused"
            )

        effect = self._current_effect()
        if self._barrier is DurabilityBarrier.EFFECT_PREPARED:
            self._ledger.append(
                effect_id=effect.effect_id,
                kind="certification",
                outcome=ProviderOutcome.NOT_STARTED,
                idempotency_key=idempotency_key,
            )
            self._require_crash(sigkill=False)

        self._ledger.append(
            effect_id=effect.effect_id,
            kind="dispatch",
            outcome=ProviderOutcome.UNKNOWN,
            idempotency_key=idempotency_key,
        )
        if self._barrier is DurabilityBarrier.PROVIDER_DISPATCHED:
            self._require_crash(sigkill=True)

        receipt = {
            "provider_receipt_id": f"receipt-{effect.effect_id}",
            "result": {
                "committed": True,
                "idempotency_key": idempotency_key,
            },
        }
        self._ledger.append(
            effect_id=effect.effect_id,
            kind="receipt",
            outcome=ProviderOutcome.SUCCEEDED,
            idempotency_key=idempotency_key,
            receipt=receipt,
        )
        return self._transport_result(request.get("id"), receipt)

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification:
        return ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
            rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
            state_mutation=True,
            information_flow=True,
            metadata={
                "operation": operation,
                "endpoint_id": context.get("endpoint_id"),
                "status": result.get("status") if isinstance(result, dict) else None,
            },
        )

    def reconcile_external_effect(self, effect: Any) -> dict[str, Any]:
        outcome = self._ledger.classify(effect.effect_id)
        if outcome is ProviderOutcome.NOT_STARTED:
            return {
                "state": "failed",
                "provider_receipt": {
                    "dispatch_status": "not_started",
                    "certified": True,
                    "source": "independent_fsync_provider_ledger",
                },
            }
        if outcome is ProviderOutcome.SUCCEEDED:
            matching = [
                row
                for row in self._ledger.records()
                if row["effect_id"] == effect.effect_id and row["kind"] == "receipt"
            ]
            if len(matching) == 1 and isinstance(matching[0]["receipt"], dict):
                return {
                    "state": "succeeded",
                    "provider_receipt": dict(matching[0]["receipt"]),
                }
        return {"state": "unknown"}

    def verify_reopen_dedupe(self, idempotency_key: str) -> bool:
        before = self._ledger.records()
        request = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "reopen-dedupe-probe",
                "method": "durability.commit",
                "params": {"idempotency_key": idempotency_key},
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        result = self.call(None, None, request)
        after = self._ledger.records()
        return (
            result.status_code == 200
            and result.error is None
            and before == after
            and self._ledger.successful_receipt_for_key(idempotency_key) is not None
        )

    def _current_effect(self) -> Any:
        if self._runtime is None or self._pid is None:
            raise AssertionError("new provider dispatch requires an attached Runtime")
        matching = [
            effect
            for effect in self._runtime.store.list_external_effects(pid=self._pid)
            if effect.provider == "jsonrpc" and effect.operation == "call"
        ]
        if not matching:
            raise AssertionError("provider dispatch has no prepared local effect")
        selected = matching[-1]
        if selected.transaction_state != "dispatched":
            raise AssertionError("provider call did not cross the durable dispatch fence")
        return selected

    @staticmethod
    def _transport_result(request_id: Any, receipt: dict[str, Any]) -> JsonRpcTransportResult:
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": dict(receipt["result"]),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return JsonRpcTransportResult(
            status_code=200,
            body=body,
            elapsed_s=0.001,
            response_bytes=len(body),
        )

    def _require_crash(self, *, sigkill: bool) -> None:
        if self._crash is None:
            raise AssertionError("crash callback is unavailable")
        self._crash(sigkill)


def run_crash_matrix(
    directory: str | os.PathLike[str],
    *,
    python_executable: str = sys.executable,
) -> tuple[CrashMatrixResult, ...]:
    """Execute every barrier in an isolated worker and verify provider truth."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    results: list[CrashMatrixResult] = []
    for barrier in DurabilityBarrier:
        recovery_class = _recovery_class_for_barrier(barrier)
        ledger_path = root / f"{barrier.value}.jsonl"
        database_path = root / f"{barrier.value}.sqlite"
        completed = subprocess.run(
            [
                python_executable,
                "-m",
                "benchmarks.durable_task_runs.crash_worker",
                "--ledger",
                str(ledger_path),
                "--database",
                str(database_path),
                "--barrier",
                barrier.value,
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        ledger = FsyncProviderLedger(ledger_path)
        provider_records = ledger.records()
        effect_id = (
            str(provider_records[0]["effect_id"])
            if provider_records
            else None
        )
        recovered = _reopen_runtime_evidence(
            database_path,
            barrier,
            effect_id,
            ledger_path,
        )
        repeated = _reopen_runtime_evidence(
            database_path,
            barrier,
            effect_id,
            ledger_path,
        )
        results.append(
            CrashMatrixResult(
                barrier=barrier,
                recovery_class=recovery_class,
                process_returncode=completed.returncode,
                provider_outcome=(
                    ledger.classify(effect_id)
                    if effect_id is not None
                    else ProviderOutcome.ABSENT
                ),
                dispatch_count=(
                    ledger.count(effect_id, "dispatch")
                    if effect_id is not None
                    else 0
                ),
                receipt_count=(
                    ledger.count(effect_id, "receipt")
                    if effect_id is not None
                    else 0
                ),
                settlement_reopen_stable=(
                    repeated["reopen_evidence_fingerprint"]
                    == recovered["reopen_evidence_fingerprint"]
                ),
                **recovered,
            )
        )
    return tuple(results)


def run_unpaired_committed_result_scenario(
    directory: str | os.PathLike[str],
    *,
    python_executable: str = sys.executable,
) -> CrashMatrixResult:
    """Crash after a committed effect but before its complete local result.

    The worker first commits an older safe point.  The provider then commits
    exactly once, but the successor has no complete paired result with which
    to advance that point.  Even authoritative provider success therefore
    remains non-replayable and requires attention.
    """

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "unpaired-committed-result.jsonl"
    database_path = root / "unpaired-committed-result.sqlite"
    completed = subprocess.run(
        [
            python_executable,
            "-m",
            "benchmarks.durable_task_runs.crash_worker",
            "--ledger",
            str(ledger_path),
            "--database",
            str(database_path),
            "--barrier",
            DurabilityBarrier.PROVIDER_RESULT_DURABLE.value,
            "--crash-before-local-result",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    ledger = FsyncProviderLedger(ledger_path)
    provider_records = ledger.records()
    effect_id = (
        str(provider_records[0]["effect_id"])
        if provider_records
        else None
    )
    recovered = _reopen_runtime_evidence(
        database_path,
        DurabilityBarrier.PROVIDER_RESULT_DURABLE,
        effect_id,
        ledger_path,
    )
    repeated = _reopen_runtime_evidence(
        database_path,
        DurabilityBarrier.PROVIDER_RESULT_DURABLE,
        effect_id,
        ledger_path,
    )
    return CrashMatrixResult(
        barrier=DurabilityBarrier.PROVIDER_RESULT_DURABLE,
        recovery_class=RecoveryClass.UNKNOWN_EFFECT,
        process_returncode=completed.returncode,
        provider_outcome=(
            ledger.classify(effect_id)
            if effect_id is not None
            else ProviderOutcome.ABSENT
        ),
        dispatch_count=(
            ledger.count(effect_id, "dispatch") if effect_id is not None else 0
        ),
        receipt_count=(
            ledger.count(effect_id, "receipt") if effect_id is not None else 0
        ),
        settlement_reopen_stable=(
            repeated["reopen_evidence_fingerprint"]
            == recovered["reopen_evidence_fingerprint"]
        ),
        **recovered,
    )


def _reopen_runtime_evidence(
    database_path: Path,
    barrier: DurabilityBarrier,
    effect_id: str | None,
    ledger_path: Path,
) -> dict[str, Any]:
    config = replace(
        DEFAULT_CONFIG,
        task_runs=replace(
            DEFAULT_CONFIG.task_runs,
            plaintext_payloads_enabled=True,
        ),
    )
    workspace = database_path.parent / f"{database_path.stem}-workspace"
    workspace.mkdir(exist_ok=True)
    provider_ledger = FsyncProviderLedger(ledger_path)
    provider = FsyncIdempotentJsonRpcProvider(provider_ledger)
    substrate = LocalResourceProviderSubstrate(workspace)
    substrate.jsonrpc = provider
    runtime = Runtime.open(database_path, config=config, substrate=substrate)
    try:
        page = runtime.task_runs.list(limit=10)
        if len(page.records) != 1:
            raise AssertionError(
                f"crash successor found {len(page.records)} TaskRuns; expected one"
            )
        summary = page.records[0]
        root = runtime.store.get_process(summary.root_pid or "")
        effect = (
            runtime.store.get_external_effect(effect_id)
            if effect_id is not None
            else None
        )
        ledger = runtime.store.list_task_run_ledger(
            summary.run_id,
            after=None,
            limit=100,
        )
        validated_action = any(
            item.kind.value == "llm_turn" and item.status == "validated"
            for item in ledger.records
        )
        tool_call_present = any(
            item.kind.value == "tool_call" for item in ledger.records
        )
        effect_link_present = effect_id is not None and any(
            link.evidence_type == "external_effect"
            and link.evidence_id == effect_id
            for link in runtime.store.list_task_run_links(summary.run_id)
        )
        resume = (
            runtime.store.get_task_run_resume_point(summary.root_pid, complete_only=True)
            if summary.root_pid is not None
            else None
        )
        idempotency_keys = {
            str(row["idempotency_key"])
            for row in provider_ledger.records()
            if isinstance(row.get("idempotency_key"), str)
            and row["idempotency_key"]
            and row["outcome"] == ProviderOutcome.SUCCEEDED.value
        }
        idempotency_dedupe_verified = False
        if len(idempotency_keys) == 1:
            idempotency_dedupe_verified = provider.verify_reopen_dedupe(
                next(iter(idempotency_keys))
            )
        # Runtime.open performs recovery/projection but never grants a dispatch
        # scope. The successor therefore proves classification without model or
        # provider execution.
        if summary.status is TaskRunStatus.NEEDS_ATTENTION and any(
            isinstance(blocker, dict) and blocker.get("kind") == "unknown_effect"
            for blocker in summary.blockers
        ):
            try:
                runtime.task_runs.run_until_blocked(
                    summary.run_id,
                    expected_revision=summary.revision,
                    command_id="successor-must-not-run-unknown",
                    max_quanta=1,
                )
            except Exception:
                pass
            refreshed = runtime.task_runs.get(summary.run_id)
            if refreshed.status is not TaskRunStatus.NEEDS_ATTENTION:
                raise AssertionError("unknown effect became dispatchable after reopen")
            summary = refreshed
        evidence_fingerprint = _reopen_evidence_fingerprint(
            runtime,
            summary.run_id,
            summary.root_pid,
        )
        return {
            "runtime_reopened": True,
            "recovered_status": summary.status.value,
            "blocker_kinds": tuple(
                sorted(
                    str(blocker.get("kind"))
                    for blocker in summary.blockers
                    if isinstance(blocker, dict)
                )
            ),
            "local_effect_transaction_state": (
                effect.transaction_state if effect is not None else None
            ),
            "root_present": root is not None,
            "validated_action_present": validated_action,
            "tool_call_present": tool_call_present,
            "effect_link_present": effect_link_present,
            "resume_point_present": resume is not None and resume.complete,
            "pending_action_present": (
                resume is not None and resume.pending_action_payload_id is not None
            ),
            "local_llm_call_count": len(
                runtime.store.list_llm_calls(pid=summary.root_pid)
            ),
            "completed_step_count": summary.completed_step_count,
            "idempotency_dedupe_verified": idempotency_dedupe_verified,
            "reopen_evidence_fingerprint": evidence_fingerprint,
        }
    finally:
        runtime.close()


def _reopen_evidence_fingerprint(
    runtime: Runtime,
    run_id: str,
    root_pid: str | None,
) -> str:
    """Hash all evidence whose cardinality/content must be reopen-idempotent.

    Runtime epoch and recovery-observation rows legitimately advance at each
    exclusive writer acquisition.  Provider effects, effect transitions,
    action/settlement ledger rows, links, payload bindings, local LLM calls,
    and the complete resume bundle must not change on a second reopen.
    """

    evidence = _reopen_evidence_projection(runtime, run_id, root_pid)
    encoded = json.dumps(
        evidence,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reopen_evidence_projection(
    runtime: Runtime,
    run_id: str,
    root_pid: str | None,
) -> dict[str, Any]:
    effects = (
        runtime.store.list_external_effects(pid=root_pid)
        if root_pid is not None
        else []
    )
    effect_ids = {effect.effect_id for effect in effects}
    transitions = [
        dict(row)
        for row in runtime.store._query(  # noqa: SLF001 - benchmark contract probe
            "SELECT seq, effect_id, effect_state, transaction_state, occurred_at "
            "FROM external_effect_transitions ORDER BY seq"
        )
        if str(row["effect_id"]) in effect_ids
    ]
    task_ledger: list[Any] = []
    cursor: int | None = None
    while True:
        page = runtime.store.list_task_run_ledger(
            run_id,
            after=cursor,
            limit=500,
        )
        task_ledger.extend(page.records)
        if page.next_cursor is None:
            break
        if not page.records or page.next_cursor == cursor:
            raise AssertionError("TaskRun ledger pagination did not advance")
        cursor = page.next_cursor
    stable_ledger = [
        item
        for item in task_ledger
        if not (
            item.kind.value == "status_transition"
            and item.status in {"runtime_claimed", "recovered"}
        )
    ]
    process = runtime.store.get_process(root_pid or "")
    return {
        "schema_version": 1,
        "run_id": run_id,
        "process": (
            {
                "pid": process.pid,
                "status": process.status.value,
                "task_run_id": process.task_run_id,
                "task_run_role": process.task_run_role,
                "resource_usage": to_jsonable(process.resource_usage),
            }
            if process is not None
            else None
        ),
        "resume_point": to_jsonable(
            runtime.store.get_task_run_resume_point(root_pid)
            if root_pid is not None
            else None
        ),
        "payloads": to_jsonable(runtime.store.list_task_run_payloads(run_id)),
        "ledger": to_jsonable(stable_ledger),
        "links": to_jsonable(runtime.store.list_task_run_links(run_id)),
        "llm_calls": to_jsonable(
            runtime.store.list_llm_calls(pid=root_pid)
            if root_pid is not None
            else []
        ),
        "effects": to_jsonable(effects),
        "effect_transitions": transitions,
    }


def _recovery_class_for_barrier(barrier: DurabilityBarrier) -> RecoveryClass:
    if barrier in {
        DurabilityBarrier.RUN_COMMITTED,
        DurabilityBarrier.ACTION_COMMITTED,
    }:
        return RecoveryClass.PURE
    if barrier is DurabilityBarrier.EFFECT_PREPARED:
        return RecoveryClass.CERTIFIED_NOT_STARTED
    if barrier in {
        DurabilityBarrier.PROVIDER_RESULT_DURABLE,
        DurabilityBarrier.RESUME_POINT_COMMITTED,
    }:
        return RecoveryClass.PROVIDER_IDEMPOTENT
    return RecoveryClass.UNKNOWN_EFFECT


def _expected_returncode(barrier: DurabilityBarrier) -> int:
    if barrier is DurabilityBarrier.PROVIDER_DISPATCHED:
        return -signal.SIGKILL
    return CRASH_EXIT_CODE


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key, value in pairs:
        if key in selected:
            raise ValueError("provider ledger contains a duplicate JSON key")
        selected[key] = value
    return selected


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"provider ledger contains non-finite JSON number: {value}")


def _fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
