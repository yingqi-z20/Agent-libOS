from __future__ import annotations

import asyncio
import contextlib
import os
import threading
from collections.abc import Iterable, Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest

from agent_libos.capability import CapabilityDraft
from agent_libos.config import AgentLibOSConfig, RuntimeDefaults
from agent_libos.models import (
    AgentImage,
    CapabilityEffect,
    CapabilityRight,
    ObjectType,
    OperationOutcome,
    ValidationResult,
)
from agent_libos.models.exceptions import (
    ProcessError,
    RuntimePublicationPending,
    RuntimeRecoveryRequired,
    ValidationError,
)
from agent_libos.runtime.runtime import Runtime
from agent_libos.storage import open_store
from agent_libos.substrate import LocalResourceProviderSubstrate, SubprocessLimits
from agent_libos.tools.sandbox import DenoTypescriptSandbox
from agent_libos.utils.serde import dumps


PERSISTENT_BACKENDS = [
    "sqlite-file",
    pytest.param("postgres", marks=pytest.mark.postgres),
]
THREAD_SYNC_TIMEOUT_S = 30.0


def _release_fenced_runtime_or_close(runtime: Runtime) -> None:
    reason = runtime.lifecycle.shutdown_reason
    if (
        runtime.lifecycle.state == "close_failed"
        and isinstance(reason, str)
        and reason.startswith("runtime.recovery_required:")
    ):
        result = runtime.release_recovery_diagnostics()
        assert result["ok"] is True, result
        assert result["recovery_diagnostics_released"] is True
        return
    runtime.close()


class _AcceptingValidationSandbox(DenoTypescriptSandbox):
    def run_tests(
        self,
        source_code: str,
        tests: list[dict[str, object]],
        timeout: float | None = None,
        *,
        limits: SubprocessLimits | None = None,
        return_metrics: bool = False,
    ) -> ValidationResult:
        del source_code, tests, timeout, limits, return_metrics
        return ValidationResult(ok=True, metadata={"language": "typescript"})


class _SimulatedCrash(BaseException):
    """Escape ordinary compensation while leaving a durable publication."""


def _abort_before_launch_compensation(
    _publication_id: str,
    _pid: str,
    error: BaseException,
) -> str:
    """Model process death at the boundary before compensation can execute."""

    raise error


def _abort_before_exec_compensation(
    *,
    error: BaseException,
    **_kwargs: object,
) -> None:
    """Model host process loss after exec effects but before rollback starts."""

    raise error


def _group_leaf_exceptions(error: BaseExceptionGroup) -> list[BaseException]:
    leaves: list[BaseException] = []
    pending: list[BaseException] = list(error.exceptions)
    while pending:
        current = pending.pop()
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        else:
            leaves.append(current)
    return leaves


def _invoke_test_launch(
    runtime: Runtime,
    launch_kind: str,
    parent_pid: str | None,
) -> str:
    if launch_kind == "spawn":
        return runtime.process.spawn(goal="launch rollback control-flow probe")
    assert parent_pid is not None
    if launch_kind == "fork":
        return runtime.process.fork(
            parent_pid,
            "fork rollback control-flow probe",
        )
    if launch_kind == "spawn_child":
        return runtime.process.spawn_child(
            parent_pid,
            "spawn-child rollback control-flow probe",
        )
    raise AssertionError(f"unknown launch kind: {launch_kind}")


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_recovery_claim_has_one_winner_and_durable_manual_attempt_limit(
    kind: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        # This is a storage-level lease/attempt contract.  Use a canonical
        # publication kind, but reopen the store directly so the runtime's
        # kind-specific startup reconciler does not consume the synthetic
        # publication between the durability assertions below.
        first_store = open_store(target, config=config)
        publication_id = f"publication-claim-contract-{uuid4().hex}"
        try:
            initial = first_store.insert_runtime_publication(
                publication_id=publication_id,
                kind="process_launch",
                pid=f"pid-claim-contract-{uuid4().hex}",
                owner_instance_id="runtime-that-crashed",
                plan={"contract": "publication recovery claim"},
            )
            barrier = threading.Barrier(3)
            outcomes: list[dict[str, object] | None] = []
            errors: list[BaseException] = []

            def claim(claimant: str) -> None:
                try:
                    barrier.wait(timeout=10)
                    outcomes.append(
                        first_store.claim_runtime_publication_recovery(
                            publication_id,
                            claimant_instance_id=claimant,
                            expected_owner_instance_id=initial["owner_instance_id"],
                            expected_state=initial["state"],
                            classification="contract_compensation",
                            max_attempts=2,
                        )
                    )
                except BaseException as exc:  # pragma: no cover - diagnostic path
                    errors.append(exc)

            threads = [
                threading.Thread(target=claim, args=("recovery-a",)),
                threading.Thread(target=claim, args=("recovery-b",)),
            ]
            for thread in threads:
                thread.start()
            barrier.wait(timeout=10)
            for thread in threads:
                thread.join(timeout=15)

            assert not errors
            assert all(not thread.is_alive() for thread in threads)
            winners = [outcome for outcome in outcomes if outcome is not None]
            assert len(winners) == 1
            assert len(outcomes) == 2
            first_claim = winners[0]
            recovery = dict(first_claim["receipt"]["recovery"])
            assert recovery["attempt"] == 1
            assert recovery["classification"] == "contract_compensation"
            assert first_claim["state"] == "rollback_pending"
            resumed = first_store.claim_runtime_publication_recovery(
                publication_id,
                claimant_instance_id=str(recovery["claimant_instance_id"]),
                expected_owner_instance_id=first_claim["owner_instance_id"],
                expected_state=first_claim["state"],
                classification="contract_compensation",
                max_attempts=2,
            )
            assert resumed is not None
            assert resumed["receipt"]["recovery"] == recovery
            assert len(
                [
                    phase
                    for phase in resumed["receipt"]["phases"]
                    if phase.get("phase") == "recovery_claimed"
                ]
            ) == 1
            assert first_store.advance_runtime_publication(
                publication_id,
                state="failed",
                phase="contract_attempt_1_failed",
                expected_states={"rollback_pending"},
                recovery_lease_id=str(recovery["lease_id"]),
            )
        finally:
            first_store.close()

        reopened_store = open_store(target, config=config)
        try:
            failed = reopened_store.get_runtime_publication(publication_id)
            assert failed is not None
            assert failed["state"] == "failed"
            assert failed["receipt"]["recovery"]["attempt"] == 1
            second_claim = reopened_store.claim_runtime_publication_recovery(
                publication_id,
                claimant_instance_id="recovery-c",
                expected_owner_instance_id=failed["owner_instance_id"],
                expected_state=failed["state"],
                classification="contract_compensation",
                max_attempts=2,
            )
            assert second_claim is not None
            second_recovery = dict(second_claim["receipt"]["recovery"])
            assert second_recovery["attempt"] == 2
            assert reopened_store.advance_runtime_publication(
                publication_id,
                state="failed",
                phase="contract_attempt_2_failed",
                expected_states={"rollback_pending"},
                recovery_lease_id=str(second_recovery["lease_id"]),
            )
        finally:
            reopened_store.close()

        exhausted_store = open_store(target, config=config)
        try:
            failed = exhausted_store.get_runtime_publication(publication_id)
            assert failed is not None
            manual = exhausted_store.claim_runtime_publication_recovery(
                publication_id,
                claimant_instance_id="recovery-d",
                expected_owner_instance_id=failed["owner_instance_id"],
                expected_state=failed["state"],
                classification="contract_compensation",
                max_attempts=2,
            )
            assert manual is not None
            assert manual["state"] == "manual"
            assert manual["phase"] == "recovery_attempts_exhausted"
            assert manual["receipt"]["recovery"] == {
                "attempt": 3,
                "classification": "contract_compensation",
                "claimant_instance_id": "recovery-d",
                "lease_id": manual["receipt"]["recovery"]["lease_id"],
                "disposition": "manual",
            }
        finally:
            exhausted_store.close()

        durable_store = open_store(target, config=config)
        try:
            durable = durable_store.get_runtime_publication(publication_id)
            assert durable is not None
            assert durable["state"] == "manual"
            claims = [
                phase
                for phase in durable["receipt"]["phases"]
                if phase.get("phase") == "recovery_claimed"
            ]
            assert [phase["attempt"] for phase in claims] == [1, 2, 3]
            assert claims[-1]["disposition"] == "manual"
            assert (
                durable_store.claim_runtime_publication_recovery(
                    publication_id,
                    claimant_instance_id="recovery-must-not-steal-manual",
                    expected_owner_instance_id=durable["owner_instance_id"],
                    expected_state=durable["state"],
                    classification="contract_compensation",
                    max_attempts=2,
                )
                is None
            )
        finally:
            durable_store.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_interrupted_exec_reopen_compensates_only_exact_owned_artifacts_once(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        substrate_root = tmp_path / f"substrate-{kind}"
        package_root = _write_image_package(substrate_root / "package-agent")
        runtime = Runtime.open(
            target,
            config=config,
            substrate=LocalResourceProviderSubstrate(substrate_root),
        )
        publication_id = ""
        pid = ""
        unowned_capability_id = ""
        original_artifacts: list[dict[str, object]] = []
        owned_capability_ids: set[str] = set()
        owned_tool_ids: set[str] = set()
        owned_candidate_ids: set[str] = set()
        owned_workspace_paths: set[str] = set()
        try:
            runtime.tools.sandbox = _AcceptingValidationSandbox()
            runtime.image_registry.register_from_package_path(package_root, actor="test")
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="interrupted exact publication ownership contract",
            )
            runtime.capability.grant(
                pid,
                runtime.image_registry.resource_for("package-agent:v0"),
                [CapabilityRight.READ],
                issued_by="test",
            )
            original_configure_skills = runtime.image_boot._configure_skills

            def crash_after_package_artifacts(*args: object, **kwargs: object) -> None:
                nonlocal unowned_capability_id
                original_configure_skills(*args, **kwargs)
                unrelated = runtime.capability.issue_trusted(
                    pid,
                    "custom:unowned-image-prefixed-authority",
                    [CapabilityRight.READ],
                    issued_by="image:independent-controller",
                )
                unowned_capability_id = unrelated.cap_id
                raise _SimulatedCrash("crash after package publication effects")

            monkeypatch.setattr(
                runtime.image_boot,
                "_configure_skills",
                crash_after_package_artifacts,
            )
            monkeypatch.setattr(
                runtime.image_boot,
                "_rollback_failed_exec",
                _abort_before_exec_compensation,
            )
            with pytest.raises(BaseExceptionGroup) as caught:
                runtime.exec_process(pid, "package-agent:v0")
            assert any(
                isinstance(item, _SimulatedCrash)
                for item in _group_leaf_exceptions(caught.value)
            )

            publications = [
                publication
                for publication in runtime.store.list_runtime_publications(pid=pid)
                if publication["kind"] == "process_exec"
            ]
            assert len(publications) == 1
            publication = publications[0]
            publication_id = publication["publication_id"]
            assert publication["state"] == "applying"
            original_artifacts = list(publication["receipt"]["artifacts"])
            kinds = {artifact["kind"] for artifact in original_artifacts}
            assert {"workspace", "capability", "tool_candidate", "tool"} <= kinds
            owned_capability_ids = {
                str(artifact["capability_id"])
                for artifact in original_artifacts
                if artifact["kind"] == "capability"
            }
            owned_tool_ids = {
                str(artifact["tool_id"])
                for artifact in original_artifacts
                if artifact["kind"] == "tool"
            }
            owned_candidate_ids = {
                str(artifact["candidate_id"])
                for artifact in original_artifacts
                if artifact["kind"] == "tool_candidate"
            }
            owned_workspace_paths = {
                str(artifact["path"])
                for artifact in original_artifacts
                if artifact["kind"] == "workspace"
            }
            candidate_positions = [
                index
                for index, artifact in enumerate(original_artifacts)
                if artifact["kind"] == "tool_candidate"
            ]
            tool_positions = [
                index
                for index, artifact in enumerate(original_artifacts)
                if artifact["kind"] == "tool"
            ]
            assert candidate_positions and tool_positions
            assert max(candidate_positions) < min(tool_positions)
            assert all(
                runtime.store.get_capability(capability_id) is not None
                for capability_id in owned_capability_ids
            )
            assert owned_tool_ids <= {
                str(row["tool_id"]) for row in runtime.store.list_tools()
            }
            assert all(
                runtime.store.get_tool_candidate(candidate_id) is not None
                for candidate_id in owned_candidate_ids
            )
            assert all(
                (substrate_root / workspace_path).exists()
                for workspace_path in owned_workspace_paths
            )
            unrelated = runtime.store.get_capability(unowned_capability_id)
            assert unrelated is not None and unrelated.active
            assert unowned_capability_id not in {
                str(artifact.get("capability_id") or "")
                for artifact in original_artifacts
            }
        finally:
            _release_fenced_runtime_or_close(runtime)

        first_reopen = Runtime.open(
            target,
            config=config,
            substrate=LocalResourceProviderSubstrate(substrate_root),
        )
        try:
            publication = first_reopen.store.get_runtime_publication(publication_id)
            assert publication is not None
            assert publication["state"] == "rolled_back"
            assert publication["phase"] == "startup_compensated"
            assert publication["receipt"]["artifacts"] == original_artifacts
            assert publication_id in first_reopen.recovered_exec_publications
            assert first_reopen.process.get(pid).image_id == "base-agent:v0"
            unrelated = first_reopen.store.get_capability(unowned_capability_id)
            assert unrelated is not None and unrelated.active
            assert unrelated.issued_by == "image:independent-controller"
            _assert_owned_artifacts_absent(
                first_reopen,
                substrate_root=substrate_root,
                capability_ids=owned_capability_ids,
                tool_ids=owned_tool_ids,
                candidate_ids=owned_candidate_ids,
                workspace_paths=owned_workspace_paths,
            )
        finally:
            first_reopen.close()

        second_reopen = Runtime.open(
            target,
            config=config,
            substrate=LocalResourceProviderSubstrate(substrate_root),
        )
        try:
            publication = second_reopen.store.get_runtime_publication(publication_id)
            assert publication is not None
            assert publication["state"] == "rolled_back"
            assert publication["phase"] == "startup_compensated"
            assert publication["receipt"]["artifacts"] == original_artifacts
            assert publication_id not in second_reopen.recovered_exec_publications
            unrelated = second_reopen.store.get_capability(unowned_capability_id)
            assert unrelated is not None and unrelated.active
            _assert_owned_artifacts_absent(
                second_reopen,
                substrate_root=substrate_root,
                capability_ids=owned_capability_ids,
                tool_ids=owned_tool_ids,
                candidate_ids=owned_candidate_ids,
                workspace_paths=owned_workspace_paths,
            )
        finally:
            second_reopen.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_launch_capability_effect_and_exact_receipt_share_one_uow_across_crash(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        publication_id = ""
        pid = ""
        receipt_ids: set[str] = set()
        try:
            original = runtime.process._record_launch_capability_artifacts
            record_calls = 0

            def crash_before_authority_receipt(
                *args: object,
                **kwargs: object,
            ) -> None:
                nonlocal record_calls
                record_calls += 1
                if record_calls == 2:
                    raise _SimulatedCrash(
                        "crash between launch authority effect and exact receipt"
                    )
                original(*args, **kwargs)

            monkeypatch.setattr(
                runtime.process,
                "_record_launch_capability_artifacts",
                crash_before_authority_receipt,
            )
            monkeypatch.setattr(
                runtime.process,
                "_finish_failed_launch",
                _abort_before_launch_compensation,
            )
            with pytest.raises(_SimulatedCrash):
                runtime.process.spawn(
                    goal="atomic launch authority receipt",
                    capabilities=[
                        {
                            "resource": "custom:launch-authority-uow",
                            "rights": [CapabilityRight.READ.value],
                        }
                    ],
                )

            publication = runtime.store.list_runtime_publications()[-1]
            publication_id = str(publication["publication_id"])
            pid = str(publication["pid"])
            assert publication["kind"] == "process_launch"
            assert publication["state"] == "applying"
            assert not [
                capability
                for capability in runtime.store.list_capabilities(pid)
                if capability.resource == "custom:launch-authority-uow"
            ]
            receipt_ids = {
                str(artifact["capability_id"])
                for artifact in publication["receipt"]["artifacts"]
                if artifact.get("kind") == "capability"
            }
            assert receipt_ids == {
                capability.cap_id
                for capability in runtime.store.list_capabilities(pid)
            }
        finally:
            runtime.close()

        reopened = Runtime.open(target, config=config)
        try:
            publication = reopened.store.get_runtime_publication(publication_id)
            assert publication is not None
            assert publication["state"] == "rolled_back"
            assert publication["phase"] == "startup_compensated"
            assert reopened.store.get_process(pid) is None
            assert reopened.store.list_capabilities(pid) == []
            assert {
                str(artifact["capability_id"])
                for artifact in publication["receipt"]["artifacts"]
                if artifact.get("kind") == "capability"
            } == receipt_ids
            reopened.image_boot.assert_publication_artifacts_removed(publication)
        finally:
            reopened.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
@pytest.mark.parametrize("transition_failure", ["cas_false", "store_error"])
def test_launch_rollback_transition_failure_fences_partial_package_until_reopen(
    kind: str,
    transition_failure: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lost rollback CAS cannot leave partial launch effects mutable."""

    with _persistent_target(kind, tmp_path) as (target, config):
        substrate_root = tmp_path / f"launch-transition-{kind}-{transition_failure}"
        package_root = _write_image_package(substrate_root / "package-agent")
        runtime = Runtime.open(
            target,
            config=config,
            substrate=LocalResourceProviderSubstrate(substrate_root),
        )
        publication_id = ""
        operation_id = ""
        pid = ""
        capability_ids: set[str] = set()
        tool_ids: set[str] = set()
        candidate_ids: set[str] = set()
        workspace_paths: set[str] = set()
        try:
            runtime.tools.sandbox = _AcceptingValidationSandbox()
            runtime.image_registry.register_from_package_path(package_root, actor="test")
            before_publication_ids = {
                str(publication["publication_id"])
                for publication in runtime.store.list_runtime_publications()
            }
            original_audit_record = runtime.audit.record
            original_advance = runtime.store.advance_runtime_publication
            rollback_attempts = 0

            def fail_package_authority_audit(
                *args: object,
                **kwargs: object,
            ) -> object:
                if kwargs.get("action") == "image.required_capabilities_declared_only":
                    raise RuntimeError("injected partial package launch audit failure")
                return original_audit_record(*args, **kwargs)

            def fail_initial_rollback_transition(
                selected_publication_id: str,
                *,
                state: str,
                phase: str,
                **kwargs: object,
            ) -> bool:
                nonlocal rollback_attempts
                if state == "rollback_pending" and phase == "compensating":
                    rollback_attempts += 1
                    if transition_failure == "cas_false":
                        return False
                    assert original_advance(
                        selected_publication_id,
                        state=state,
                        phase=phase,
                        **kwargs,
                    )
                    raise RuntimeError("injected rollback transition store failure")
                return original_advance(
                    selected_publication_id,
                    state=state,
                    phase=phase,
                    **kwargs,
                )

            monkeypatch.setattr(runtime.audit, "record", fail_package_authority_audit)
            monkeypatch.setattr(
                runtime.store,
                "advance_runtime_publication",
                fail_initial_rollback_transition,
            )

            with pytest.raises(RuntimePublicationPending) as caught:
                runtime.process.spawn(
                    image="package-agent:v0",
                    goal="fence a partial package launch",
                )

            assert rollback_attempts == 1
            publications = [
                publication
                for publication in runtime.store.list_runtime_publications()
                if publication["kind"] == "process_launch"
                and publication["publication_id"] not in before_publication_ids
            ]
            assert len(publications) == 1
            publication = publications[0]
            publication_id = str(publication["publication_id"])
            operation_id = str(publication["plan"]["operation_id"])
            pid = str(publication["pid"])
            assert caught.value.publication_id == publication_id
            assert caught.value.operation_id == operation_id
            if transition_failure == "cas_false":
                assert publication["state"] == "applying"
                assert publication["phase"] != "compensating"
            else:
                # The rollback CAS and launch-goal redaction share one outer
                # transaction. A storage exception before that boundary
                # acknowledges commit must therefore restore the earlier
                # recoverable state rather than publish rollback_pending.
                assert publication["state"] == "applying"
                assert publication["phase"] != "compensating"
            assert runtime.lifecycle.state == "close_failed"
            assert runtime.lifecycle.shutdown_reason == (
                f"runtime.recovery_required:{publication_id}"
            )

            operation = runtime.store.get_operation(operation_id)
            assert operation is not None
            assert operation.state.value == "running"
            assert operation.outcome.value == "pending"
            assert operation.metadata["runtime_publication_id"] == publication_id
            assert operation.metadata["runtime_publication_bound"] is True

            artifacts = list(publication["receipt"]["artifacts"])
            assert {
                "workspace",
                "capability",
                "tool_candidate",
                "tool",
            } <= {str(artifact["kind"]) for artifact in artifacts}
            capability_ids = {
                str(artifact["capability_id"])
                for artifact in artifacts
                if artifact["kind"] == "capability"
            }
            tool_ids = {
                str(artifact["tool_id"])
                for artifact in artifacts
                if artifact["kind"] == "tool"
            }
            candidate_ids = {
                str(artifact["candidate_id"])
                for artifact in artifacts
                if artifact["kind"] == "tool_candidate"
            }
            workspace_paths = {
                str(artifact["path"])
                for artifact in artifacts
                if artifact["kind"] == "workspace"
            }
            assert all(
                (substrate_root / workspace_path).exists()
                for workspace_path in workspace_paths
            )
            with pytest.raises(
                RuntimeError,
                match="not accepting operations: state=close_failed",
            ):
                runtime.process.spawn(goal="must wait for launch recovery")
        finally:
            _release_fenced_runtime_or_close(runtime)

        reopened = Runtime.open(
            target,
            config=config,
            substrate=LocalResourceProviderSubstrate(substrate_root),
        )
        try:
            publication = reopened.store.get_runtime_publication(publication_id)
            assert publication is not None
            assert publication["state"] == "rolled_back"
            assert publication["phase"] == "startup_compensated"
            assert publication_id in reopened.recovered_runtime_publications
            assert reopened.store.get_process(pid) is None
            operation = reopened.store.get_operation(operation_id)
            assert operation is not None
            assert operation.state.value == "terminal"
            assert operation.outcome.value == "failed"
            assert operation.metadata["runtime_publication_id"] == publication_id
            _assert_owned_artifacts_absent(
                reopened,
                substrate_root=substrate_root,
                capability_ids=capability_ids,
                tool_ids=tool_ids,
                candidate_ids=candidate_ids,
                workspace_paths=workspace_paths,
            )
        finally:
            reopened.close()


def test_launch_rollback_cas_false_honors_a_concurrently_committed_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A durable commit wins over a stale failure path without poisoning."""

    runtime = Runtime.open("local")
    try:
        original_transaction = runtime.store.transaction
        injected = False

        @contextlib.contextmanager
        def raise_after_launch_commit(
            *,
            include_object_payloads: bool = False,
        ) -> Iterator[object]:
            nonlocal injected
            with original_transaction(
                include_object_payloads=include_object_payloads
            ) as cursor:
                yield cursor
            if not injected and any(
                publication["kind"] == "process_launch"
                and publication["state"] == "committed"
                for publication in runtime.store.list_runtime_publications()
            ):
                injected = True
                raise RuntimeError("injected post-commit launch reporting failure")

        monkeypatch.setattr(runtime.store, "transaction", raise_after_launch_commit)
        pid = runtime.process.spawn(goal="durable committed launch wins")

        assert injected
        publication = [
            item
            for item in runtime.store.list_runtime_publications(pid=pid)
            if item["kind"] == "process_launch"
        ][-1]
        assert publication["state"] == "committed"
        operation_id = str(publication["plan"]["operation_id"])
        operation = runtime.store.get_operation(operation_id)
        assert operation is not None
        assert operation.state.value == "terminal"
        assert operation.outcome.value == "succeeded"
        assert operation.pid == pid
        assert operation.metadata["runtime_publication_id"] == publication[
            "publication_id"
        ]
        assert runtime.lifecycle.state == "open"
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "interruption_kind",
    ["keyboard_interrupt", "cancelled_error"],
)
def test_committed_launch_preserves_post_commit_base_exception(
    interruption_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A committed launch stays authoritative without swallowing cancellation."""

    runtime = Runtime.open("local")
    interruption: BaseException = (
        KeyboardInterrupt("injected post-commit launch keyboard interrupt")
        if interruption_kind == "keyboard_interrupt"
        else asyncio.CancelledError("injected post-commit launch cancellation")
    )
    try:
        original_transaction = runtime.store.transaction
        injected = False

        @contextlib.contextmanager
        def interrupt_after_launch_commit(
            *,
            include_object_payloads: bool = False,
        ) -> Iterator[object]:
            nonlocal injected
            with original_transaction(
                include_object_payloads=include_object_payloads
            ) as cursor:
                yield cursor
            if not injected and any(
                publication["kind"] == "process_launch"
                and publication["state"] == "committed"
                for publication in runtime.store.list_runtime_publications()
            ):
                injected = True
                raise interruption

        monkeypatch.setattr(
            runtime.store,
            "transaction",
            interrupt_after_launch_commit,
        )
        with pytest.raises(BaseExceptionGroup) as caught:
            runtime.process.spawn(goal="preserve post-commit interruption")
        leaves = _group_leaf_exceptions(caught.value)
        assert any(item is interruption for item in leaves)
        assert any(
            isinstance(item, ProcessError)
            and "changed before compensation" in str(item)
            for item in leaves
        )
        assert injected

        publication = [
            item
            for item in runtime.store.list_runtime_publications()
            if item["kind"] == "process_launch"
        ][-1]
        pid = str(publication["pid"])
        assert publication["state"] == "committed"
        assert publication["phase"] == "committed"
        process = runtime.store.get_process(pid)
        assert process is not None
        assert process.status.value == "runnable"
        operation = runtime.store.get_operation(
            str(publication["plan"]["operation_id"])
        )
        assert operation is not None
        assert operation.state.value == "terminal"
        assert operation.outcome.value == "succeeded"
        assert operation.pid == pid
        assert operation.metadata["runtime_publication_id"] == publication[
            "publication_id"
        ]
        assert runtime.lifecycle.state == "open"
    finally:
        runtime.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
@pytest.mark.parametrize(
    "interruption_kind",
    ["keyboard_interrupt", "cancelled_error"],
)
def test_partial_package_launch_compensates_base_exception_before_propagating(
    kind: str,
    interruption_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catchable control-flow interruption cannot bypass launch rollback."""

    with _persistent_target(kind, tmp_path) as (target, config):
        substrate_root = tmp_path / f"launch-interrupt-{kind}-{interruption_kind}"
        package_root = _write_image_package(substrate_root / "package-agent")
        runtime = Runtime.open(
            target,
            config=config,
            substrate=LocalResourceProviderSubstrate(substrate_root),
        )
        publication_id = ""
        operation_id = ""
        pid = ""
        interruption: BaseException = (
            KeyboardInterrupt("injected package launch keyboard interrupt")
            if interruption_kind == "keyboard_interrupt"
            else asyncio.CancelledError("injected package launch cancellation")
        )
        try:
            runtime.tools.sandbox = _AcceptingValidationSandbox()
            runtime.image_registry.register_from_package_path(package_root, actor="test")
            before_publication_ids = {
                str(publication["publication_id"])
                for publication in runtime.store.list_runtime_publications()
            }
            original_audit_record = runtime.audit.record

            def interrupt_package_authority_audit(
                *args: object,
                **kwargs: object,
            ) -> object:
                if kwargs.get("action") == "image.required_capabilities_declared_only":
                    raise interruption
                return original_audit_record(*args, **kwargs)

            monkeypatch.setattr(
                runtime.audit,
                "record",
                interrupt_package_authority_audit,
            )
            with pytest.raises(type(interruption)) as caught:
                runtime.process.spawn(
                    image="package-agent:v0",
                    goal="compensate a catchable launch interruption",
                )
            assert caught.value is interruption

            publications = [
                publication
                for publication in runtime.store.list_runtime_publications()
                if publication["kind"] == "process_launch"
                and publication["publication_id"] not in before_publication_ids
            ]
            assert len(publications) == 1
            publication = publications[0]
            publication_id = str(publication["publication_id"])
            operation_id = str(publication["plan"]["operation_id"])
            pid = str(publication["pid"])
            assert publication["state"] == "rolled_back"
            assert publication["phase"] == "compensated"
            assert runtime.store.get_process(pid) is None
            assert runtime.lifecycle.state == "open"

            operation = runtime.store.get_operation(operation_id)
            assert operation is not None
            assert operation.state.value == "terminal"
            assert operation.outcome.value == "failed"
            assert operation.metadata["runtime_publication_id"] == publication_id

            artifacts = list(publication["receipt"]["artifacts"])
            _assert_owned_artifacts_absent(
                runtime,
                substrate_root=substrate_root,
                capability_ids={
                    str(artifact["capability_id"])
                    for artifact in artifacts
                    if artifact["kind"] == "capability"
                },
                tool_ids={
                    str(artifact["tool_id"])
                    for artifact in artifacts
                    if artifact["kind"] == "tool"
                },
                candidate_ids={
                    str(artifact["candidate_id"])
                    for artifact in artifacts
                    if artifact["kind"] == "tool_candidate"
                },
                workspace_paths={
                    str(artifact["path"])
                    for artifact in artifacts
                    if artifact["kind"] == "workspace"
                },
            )

            # Restore the injected audit sink before proving that successful
            # compensation left ordinary mutation admission available.
            monkeypatch.setattr(runtime.audit, "record", original_audit_record)
            assert runtime.process.spawn(goal="allowed after compensated interrupt")
        finally:
            runtime.close()

        reopened = Runtime.open(
            target,
            config=config,
            substrate=LocalResourceProviderSubstrate(substrate_root),
        )
        try:
            publication = reopened.store.get_runtime_publication(publication_id)
            assert publication is not None
            assert publication["state"] == "rolled_back"
            assert publication["phase"] == "compensated"
            assert publication_id not in reopened.recovered_runtime_publications
            operation = reopened.store.get_operation(operation_id)
            assert operation is not None
            assert operation.state.value == "terminal"
            assert operation.outcome.value == "failed"
            assert reopened.store.get_process(pid) is None
        finally:
            reopened.close()


@pytest.mark.parametrize(
    "interruption_point",
    ["publication_read", "operation_read", "recovery_fence"],
)
def test_launch_rollback_base_exception_diagnostic_stays_fail_closed(
    interruption_point: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second interruption cannot reopen the rollback ambiguity window."""

    target = tmp_path / f"launch-diagnostic-{interruption_point}.sqlite"
    runtime = Runtime.open(target)
    publication_id = ""
    operation_id = ""
    pid = ""
    try:
        before_publication_ids = {
            str(publication["publication_id"])
            for publication in runtime.store.list_runtime_publications()
        }
        original_advance = runtime.store.advance_runtime_publication
        original_get_publication = runtime.store.get_runtime_publication
        original_get_operation = runtime.store.get_operation
        original_recovery_fence = runtime.process._recovery_required_callback
        rollback_attempted = False
        diagnostic_interrupted = False

        def fail_launch_body(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("injected diagnostic launch body failure")

        def reject_rollback_transition(
            selected_publication_id: str,
            *,
            state: str,
            phase: str,
            **kwargs: object,
        ) -> bool:
            nonlocal rollback_attempted
            if state == "rollback_pending" and phase == "compensating":
                rollback_attempted = True
                return False
            return original_advance(
                selected_publication_id,
                state=state,
                phase=phase,
                **kwargs,
            )

        def interrupt_publication_read(
            selected_publication_id: str,
        ) -> dict[str, object] | None:
            nonlocal diagnostic_interrupted
            if rollback_attempted and not diagnostic_interrupted:
                diagnostic_interrupted = True
                raise KeyboardInterrupt("injected durable publication diagnostic")
            return original_get_publication(selected_publication_id)

        def interrupt_operation_read(selected_operation_id: str) -> object:
            nonlocal diagnostic_interrupted
            if runtime.lifecycle.state == "close_failed" and not diagnostic_interrupted:
                diagnostic_interrupted = True
                raise KeyboardInterrupt("injected operation binding diagnostic")
            return original_get_operation(selected_operation_id)

        def interrupt_recovery_fence(*, publication_id: str) -> None:
            nonlocal diagnostic_interrupted
            assert original_recovery_fence is not None
            original_recovery_fence(publication_id=publication_id)
            if not diagnostic_interrupted:
                diagnostic_interrupted = True
                raise KeyboardInterrupt("injected recovery fence interruption")

        runtime.process.add_after_spawn_hook(fail_launch_body)
        monkeypatch.setattr(
            runtime.store,
            "advance_runtime_publication",
            reject_rollback_transition,
        )
        if interruption_point == "publication_read":
            monkeypatch.setattr(
                runtime.store,
                "get_runtime_publication",
                interrupt_publication_read,
            )
        elif interruption_point == "operation_read":
            monkeypatch.setattr(
                runtime.store,
                "get_operation",
                interrupt_operation_read,
            )
        else:
            monkeypatch.setattr(
                runtime.process,
                "_recovery_required_callback",
                interrupt_recovery_fence,
            )

        with pytest.raises(BaseExceptionGroup) as caught:
            runtime.process.spawn(goal="fence interrupted launch diagnostics")

        assert rollback_attempted
        assert diagnostic_interrupted
        leaves = _group_leaf_exceptions(caught.value)
        assert any(
            isinstance(item, RuntimeError)
            and "injected diagnostic launch body failure" in str(item)
            for item in leaves
        )
        assert any(
            isinstance(item, KeyboardInterrupt) and "injected" in str(item)
            for item in leaves
        )
        assert runtime.lifecycle.state == "close_failed"

        publications = [
            publication
            for publication in runtime.store.list_runtime_publications()
            if publication["kind"] == "process_launch"
            and publication["publication_id"] not in before_publication_ids
        ]
        assert len(publications) == 1
        publication = publications[0]
        publication_id = str(publication["publication_id"])
        operation_id = str(publication["plan"]["operation_id"])
        pid = str(publication["pid"])
        assert publication["state"] == "applying"
        with pytest.raises(
            RuntimeError,
            match="not accepting operations: state=close_failed",
        ):
            runtime.process.spawn(goal="must reopen after interrupted diagnostics")
    finally:
        _release_fenced_runtime_or_close(runtime)

    reopened = Runtime.open(target)
    try:
        assert reopened.lifecycle.state == "open"
        publication = reopened.store.get_runtime_publication(publication_id)
        assert publication is not None
        assert publication["state"] == "rolled_back"
        assert publication["phase"] == "startup_compensated"
        assert publication_id in reopened.recovered_runtime_publications
        operation = reopened.store.get_operation(operation_id)
        assert operation is not None
        assert operation.state.value == "terminal"
        assert operation.outcome.value == "failed"
        assert reopened.store.get_process(pid) is None
    finally:
        reopened.close()


def test_committed_launch_propagates_secondary_transition_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Committed may absorb only an all-Exception rollback aggregate."""

    runtime = Runtime.open("local")
    primary_error = RuntimeError("injected committed post-commit failure")
    secondary_interrupt = KeyboardInterrupt(
        "injected committed rollback transition interrupt"
    )
    try:
        original_transaction = runtime.store.transaction
        original_advance = runtime.store.advance_runtime_publication
        post_commit_failed = False
        transition_interrupted = False

        @contextlib.contextmanager
        def fail_after_launch_commit(
            *,
            include_object_payloads: bool = False,
        ) -> Iterator[object]:
            nonlocal post_commit_failed
            with original_transaction(
                include_object_payloads=include_object_payloads
            ) as cursor:
                yield cursor
            if not post_commit_failed and any(
                publication["kind"] == "process_launch"
                and publication["state"] == "committed"
                for publication in runtime.store.list_runtime_publications()
            ):
                post_commit_failed = True
                raise primary_error

        def interrupt_rollback_transition(
            publication_id: str,
            *,
            state: str,
            phase: str,
            **kwargs: object,
        ) -> bool:
            nonlocal transition_interrupted
            if state == "rollback_pending" and phase == "compensating":
                transition_interrupted = True
                raise secondary_interrupt
            return original_advance(
                publication_id,
                state=state,
                phase=phase,
                **kwargs,
            )

        monkeypatch.setattr(runtime.store, "transaction", fail_after_launch_commit)
        monkeypatch.setattr(
            runtime.store,
            "advance_runtime_publication",
            interrupt_rollback_transition,
        )
        with pytest.raises(BaseExceptionGroup) as caught:
            runtime.process.spawn(goal="committed transition interruption")

        leaves = _group_leaf_exceptions(caught.value)
        assert any(item is primary_error for item in leaves)
        assert any(item is secondary_interrupt for item in leaves)
        assert post_commit_failed and transition_interrupted
        publication = [
            item
            for item in runtime.store.list_runtime_publications()
            if item["kind"] == "process_launch"
        ][-1]
        assert publication["state"] == "committed"
        pid = str(publication["pid"])
        assert runtime.store.get_process(pid) is not None
        operation = runtime.store.get_operation(
            str(publication["plan"]["operation_id"])
        )
        assert operation is not None
        assert operation.state.value == "terminal"
        assert operation.outcome.value == "succeeded"
        assert runtime.lifecycle.state == "open"
    finally:
        runtime.close()


def test_rollback_outer_transaction_propagates_complete_transition_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nested terminal write cannot escape an interrupted outer rollback."""

    runtime = Runtime.open("local")
    primary_error = RuntimeError("injected launch body before concurrent rollback")
    secondary_interrupt = KeyboardInterrupt(
        "injected interrupt after concurrent rollback"
    )
    try:
        original_advance = runtime.store.advance_runtime_publication

        def fail_launch_body(*_args: object, **_kwargs: object) -> None:
            raise primary_error

        def resolve_rollback_then_interrupt(
            publication_id: str,
            *,
            state: str,
            phase: str,
            **kwargs: object,
        ) -> bool:
            if state == "rollback_pending" and phase == "compensating":
                with runtime.store.transaction(include_object_payloads=True):
                    assert original_advance(
                        publication_id,
                        state=state,
                        phase=phase,
                        **kwargs,
                    )
                    publication = runtime.store.get_runtime_publication(
                        publication_id
                    )
                    assert publication is not None
                    runtime.process._cleanup_failed_launch_strict(publication)
                    runtime.process._terminalize_launch_publication(
                        publication_id,
                        state="rolled_back",
                        phase="concurrent_compensated",
                        outcome=OperationOutcome.FAILED,
                        receipt={
                            "phase": "concurrent_compensated",
                            "pid": publication["pid"],
                        },
                    )
                raise secondary_interrupt
            return original_advance(
                publication_id,
                state=state,
                phase=phase,
                **kwargs,
            )

        runtime.process.add_after_spawn_hook(fail_launch_body)
        monkeypatch.setattr(
            runtime.store,
            "advance_runtime_publication",
            resolve_rollback_then_interrupt,
        )
        with pytest.raises(BaseExceptionGroup) as caught:
            runtime.process.spawn(goal="concurrent rolled-back aggregate")

        leaves = _group_leaf_exceptions(caught.value)
        assert any(item is primary_error for item in leaves)
        assert any(item is secondary_interrupt for item in leaves)
        publication = [
            item
            for item in runtime.store.list_runtime_publications()
            if item["kind"] == "process_launch"
        ][-1]
        assert publication["state"] == "applying"
        assert publication["phase"] != "concurrent_compensated"
        assert runtime.store.get_process(str(publication["pid"])) is not None
        operation = runtime.store.get_operation(
            str(publication["plan"]["operation_id"])
        )
        assert operation is not None
        assert operation.state.value == "running"
        assert operation.outcome.value == "pending"
        assert runtime.lifecycle.state == "close_failed"
    finally:
        _release_fenced_runtime_or_close(runtime)


@pytest.mark.parametrize(
    ("launch_kind", "interruption_kind"),
    [("spawn", "keyboard_interrupt"), ("spawn_child", "cancelled_error")],
)
def test_terminal_publication_ack_interrupt_preserves_rolled_back_aggregate(
    launch_kind: str,
    interruption_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-commit terminal ack failure cannot downgrade control flow."""

    target = tmp_path / f"terminal-ack-{launch_kind}-{interruption_kind}.sqlite"
    runtime = Runtime.open(target)
    parent_pid = (
        runtime.process.spawn(goal="terminal ack child parent")
        if launch_kind != "spawn"
        else None
    )
    publication_id = ""
    operation_id = ""
    launched_pid = ""
    primary_error = RuntimeError(f"injected {launch_kind} launch body failure")
    ack_interrupt: BaseException = (
        KeyboardInterrupt("injected rolled-back terminal ack interrupt")
        if interruption_kind == "keyboard_interrupt"
        else asyncio.CancelledError(
            "injected rolled-back terminal ack cancellation"
        )
    )
    try:
        before_publication_ids = {
            str(publication["publication_id"])
            for publication in runtime.store.list_runtime_publications()
        }
        original_transaction = runtime.store.transaction
        acknowledgement_interrupted = False

        def fail_launch_body(*_args: object, **_kwargs: object) -> None:
            raise primary_error

        @contextlib.contextmanager
        def interrupt_after_terminal_commit(
            *,
            include_object_payloads: bool = False,
        ) -> Iterator[object]:
            nonlocal acknowledgement_interrupted
            with original_transaction(
                include_object_payloads=include_object_payloads
            ) as cursor:
                yield cursor
            if not acknowledgement_interrupted and any(
                publication["kind"] == "process_launch"
                and publication["publication_id"] not in before_publication_ids
                and publication["state"] == "rolled_back"
                and publication["phase"] == "compensated"
                for publication in runtime.store.list_runtime_publications()
            ):
                acknowledgement_interrupted = True
                raise ack_interrupt

        runtime.process.add_after_spawn_hook(fail_launch_body)
        monkeypatch.setattr(
            runtime.store,
            "transaction",
            interrupt_after_terminal_commit,
        )
        with pytest.raises(BaseExceptionGroup) as caught:
            _invoke_test_launch(runtime, launch_kind, parent_pid)

        leaves = _group_leaf_exceptions(caught.value)
        assert any(item is primary_error for item in leaves)
        assert any(item is ack_interrupt for item in leaves)
        assert acknowledgement_interrupted
        publications = [
            publication
            for publication in runtime.store.list_runtime_publications()
            if publication["kind"] == "process_launch"
            and publication["publication_id"] not in before_publication_ids
        ]
        assert len(publications) == 1
        publication = publications[0]
        publication_id = str(publication["publication_id"])
        operation_id = str(publication["plan"]["operation_id"])
        launched_pid = str(publication["pid"])
        assert publication["state"] == "rolled_back"
        assert publication["phase"] == "compensated"
        assert runtime.store.get_process(launched_pid) is None
        operation = runtime.store.get_operation(operation_id)
        assert operation is not None
        assert operation.state.value == "terminal"
        assert operation.outcome.value == "failed"
        assert runtime.lifecycle.state == "open"
        if parent_pid is not None:
            assert runtime.process.list_children(parent_pid) == []
            assert runtime.process.get(parent_pid).resource_usage.child_processes == 0
    finally:
        _release_fenced_runtime_or_close(runtime)

    reopened = Runtime.open(target)
    try:
        publication = reopened.store.get_runtime_publication(publication_id)
        assert publication is not None
        assert publication["state"] == "rolled_back"
        assert publication["phase"] == "compensated"
        assert publication_id not in reopened.recovered_runtime_publications
        operation = reopened.store.get_operation(operation_id)
        assert operation is not None
        assert operation.state.value == "terminal"
        assert operation.outcome.value == "failed"
        assert reopened.store.get_process(launched_pid) is None
    finally:
        reopened.close()


@pytest.mark.parametrize("launch_kind", ["spawn", "fork", "spawn_child"])
def test_grouped_launch_interrupt_keeps_exact_publication_operation_pending(
    launch_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OperationManager must recognize an exact pending signal inside a group."""

    target = tmp_path / f"grouped-pending-{launch_kind}.sqlite"
    runtime = Runtime.open(target)
    publication_id = ""
    operation_id = ""
    launched_pid = ""
    parent_pid = (
        runtime.process.spawn(goal=f"{launch_kind} grouped pending parent")
        if launch_kind != "spawn"
        else None
    )
    primary_error = RuntimeError(f"injected {launch_kind} launch body failure")
    secondary_interrupt = KeyboardInterrupt(
        f"injected {launch_kind} rollback transition interrupt"
    )
    try:
        before_publication_ids = {
            str(publication["publication_id"])
            for publication in runtime.store.list_runtime_publications()
        }
        original_advance = runtime.store.advance_runtime_publication

        def fail_launch_body(*_args: object, **_kwargs: object) -> None:
            raise primary_error

        def interrupt_rollback_transition(
            selected_publication_id: str,
            *,
            state: str,
            phase: str,
            **kwargs: object,
        ) -> bool:
            if state == "rollback_pending" and phase == "compensating":
                raise secondary_interrupt
            return original_advance(
                selected_publication_id,
                state=state,
                phase=phase,
                **kwargs,
            )

        runtime.process.add_after_spawn_hook(fail_launch_body)
        monkeypatch.setattr(
            runtime.store,
            "advance_runtime_publication",
            interrupt_rollback_transition,
        )
        with pytest.raises(BaseExceptionGroup) as caught:
            _invoke_test_launch(runtime, launch_kind, parent_pid)

        leaves = _group_leaf_exceptions(caught.value)
        assert any(item is primary_error for item in leaves)
        assert any(item is secondary_interrupt for item in leaves)
        pending = [
            item for item in leaves if isinstance(item, RuntimePublicationPending)
        ]
        assert len(pending) == 1
        publications = [
            publication
            for publication in runtime.store.list_runtime_publications()
            if publication["kind"] == "process_launch"
            and publication["publication_id"] not in before_publication_ids
        ]
        assert len(publications) == 1
        publication = publications[0]
        publication_id = str(publication["publication_id"])
        operation_id = str(publication["plan"]["operation_id"])
        launched_pid = str(publication["pid"])
        assert pending[0].publication_id == publication_id
        assert pending[0].operation_id == operation_id
        assert publication["state"] == "applying"
        operation = runtime.store.get_operation(operation_id)
        assert operation is not None
        assert operation.state.value == "running"
        assert operation.outcome.value == "pending"
        assert runtime.lifecycle.state == "close_failed"
    finally:
        _release_fenced_runtime_or_close(runtime)

    reopened = Runtime.open(target)
    try:
        publication = reopened.store.get_runtime_publication(publication_id)
        assert publication is not None
        assert publication["state"] == "rolled_back"
        assert publication["phase"] == "startup_compensated"
        assert publication_id in reopened.recovered_runtime_publications
        operation = reopened.store.get_operation(operation_id)
        assert operation is not None
        assert operation.state.value == "terminal"
        assert operation.outcome.value == "failed"
        assert reopened.store.get_process(launched_pid) is None
    finally:
        reopened.close()


@pytest.mark.parametrize(
    ("launch_kind", "interruption_kind"),
    [("fork", "keyboard_interrupt"), ("spawn_child", "cancelled_error")],
)
def test_child_launch_fence_fallback_survives_pre_mark_interrupt(
    launch_kind: str,
    interruption_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The builder-bound lifecycle fallback fences before-mark interruption."""

    target = tmp_path / f"fallback-{launch_kind}-{interruption_kind}.sqlite"
    runtime = Runtime.open(target)
    parent_pid = runtime.process.spawn(
        goal=f"{launch_kind} recovery-fence fallback parent"
    )
    publication_id = ""
    operation_id = ""
    child_pid = ""
    primary_error = RuntimeError(f"injected {launch_kind} launch body failure")
    fence_interrupt: BaseException = (
        KeyboardInterrupt("injected pre-mark recovery fence interrupt")
        if interruption_kind == "keyboard_interrupt"
        else asyncio.CancelledError("injected pre-mark recovery fence cancellation")
    )
    try:
        before_publication_ids = {
            str(publication["publication_id"])
            for publication in runtime.store.list_runtime_publications()
        }
        original_advance = runtime.store.advance_runtime_publication

        def fail_child_launch(*_args: object, **_kwargs: object) -> None:
            raise primary_error

        def reject_rollback_transition(
            selected_publication_id: str,
            *,
            state: str,
            phase: str,
            **kwargs: object,
        ) -> bool:
            if state == "rollback_pending" and phase == "compensating":
                return False
            return original_advance(
                selected_publication_id,
                state=state,
                phase=phase,
                **kwargs,
            )

        def interrupt_before_mark(*, publication_id: str) -> None:
            del publication_id
            raise fence_interrupt

        runtime.process.add_after_spawn_hook(fail_child_launch)
        monkeypatch.setattr(
            runtime.store,
            "advance_runtime_publication",
            reject_rollback_transition,
        )
        monkeypatch.setattr(
            runtime.process,
            "_recovery_required_callback",
            interrupt_before_mark,
        )
        with pytest.raises(BaseExceptionGroup) as caught:
            _invoke_test_launch(runtime, launch_kind, parent_pid)

        leaves = _group_leaf_exceptions(caught.value)
        assert any(item is primary_error for item in leaves)
        assert any(item is fence_interrupt for item in leaves)
        assert runtime.lifecycle.state == "close_failed"
        publications = [
            publication
            for publication in runtime.store.list_runtime_publications()
            if publication["kind"] == "process_launch"
            and publication["publication_id"] not in before_publication_ids
        ]
        assert len(publications) == 1
        publication = publications[0]
        publication_id = str(publication["publication_id"])
        operation_id = str(publication["plan"]["operation_id"])
        child_pid = str(publication["pid"])
        assert publication["state"] == "applying"
        with pytest.raises(
            RuntimeError,
            match="not accepting operations: state=close_failed",
        ):
            runtime.process.spawn(goal="must reopen after fallback fence")
    finally:
        _release_fenced_runtime_or_close(runtime)

    reopened = Runtime.open(target)
    try:
        assert reopened.lifecycle.state == "open"
        publication = reopened.store.get_runtime_publication(publication_id)
        assert publication is not None
        assert publication["state"] == "rolled_back"
        assert publication["phase"] == "startup_compensated"
        operation = reopened.store.get_operation(operation_id)
        assert operation is not None
        assert operation.state.value == "terminal"
        assert operation.outcome.value == "failed"
        assert reopened.store.get_process(child_pid) is None
        assert reopened.process.list_children(parent_pid) == []
        assert reopened.process.get(parent_pid).resource_usage.child_processes == 0
    finally:
        reopened.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_failed_exec_serializes_concurrent_unowned_candidate_and_preserves_it(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        candidate_id = ""
        descriptor_oid = ""
        try:
            runtime.tools.sandbox = _AcceptingValidationSandbox()
            runtime.register_image(
                AgentImage(
                    image_id="publication-candidate-race:v0",
                    name="publication-candidate-race",
                    default_skills=["missing-publication-candidate-race-skill"],
                ),
                actor="test",
            )
            pid = runtime.process.spawn(goal="preserve concurrent unowned candidate")
            runtime.capability.grant(
                pid,
                runtime.image_registry.resource_for("publication-candidate-race:v0"),
                [CapabilityRight.READ],
                issued_by="test",
            )

            exec_paused = threading.Event()
            release_exec = threading.Event()
            proposal_reached_lock = threading.Event()
            proposal_finished = threading.Event()
            exec_errors: list[BaseException] = []
            proposal_errors: list[BaseException] = []
            candidate_ids: list[str] = []
            original_configure_skills = runtime.image_boot._configure_skills
            original_registry_lifecycle_lock = (
                runtime.tools._registry_lifecycle_lock
            )
            proposal_thread_name = "unowned-candidate-proposal"

            def observed_registry_lifecycle_lock() -> object:
                if threading.current_thread().name == proposal_thread_name:
                    proposal_reached_lock.set()
                return original_registry_lifecycle_lock()

            def pause_before_late_failure(*args: object, **kwargs: object) -> None:
                exec_paused.set()
                if not release_exec.wait(timeout=THREAD_SYNC_TIMEOUT_S):
                    raise AssertionError("timed out waiting to release failed exec")
                original_configure_skills(*args, **kwargs)

            def fail_exec() -> None:
                try:
                    runtime.exec_process(pid, "publication-candidate-race:v0")
                except BaseException as exc:
                    exec_errors.append(exc)

            def propose_unowned_candidate() -> None:
                try:
                    candidate_ids.append(
                        runtime.tools.propose(
                            pid,
                            {
                                "name": "concurrent_unowned_candidate",
                                "description": "Must commit after failed exec rollback.",
                                "input_schema": {"type": "object"},
                                "output_schema": {"type": "object"},
                            },
                            (
                                "export async function run(args: unknown, libos: unknown) "
                                "{ return { ok: true }; }\n"
                            ),
                        )
                    )
                except BaseException as exc:
                    proposal_errors.append(exc)
                finally:
                    proposal_finished.set()

            monkeypatch.setattr(
                runtime.image_boot,
                "_configure_skills",
                pause_before_late_failure,
            )
            monkeypatch.setattr(
                runtime.tools,
                "_registry_lifecycle_lock",
                observed_registry_lifecycle_lock,
            )
            exec_thread = threading.Thread(target=fail_exec)
            proposal_thread = threading.Thread(
                target=propose_unowned_candidate,
                name=proposal_thread_name,
            )
            exec_thread.start()
            assert exec_paused.wait(timeout=THREAD_SYNC_TIMEOUT_S)
            unexpected_lock_acquisition = (
                runtime._registry_lifecycle_lock.acquire(blocking=False)
            )
            if unexpected_lock_acquisition:
                runtime._registry_lifecycle_lock.release()
            assert not unexpected_lock_acquisition
            proposal_thread.start()
            assert proposal_reached_lock.wait(timeout=THREAD_SYNC_TIMEOUT_S)
            assert not proposal_finished.is_set()
            release_exec.set()
            exec_thread.join(timeout=THREAD_SYNC_TIMEOUT_S)
            proposal_thread.join(timeout=THREAD_SYNC_TIMEOUT_S)

            assert not exec_thread.is_alive()
            assert not proposal_thread.is_alive()
            assert len(exec_errors) == 1
            assert "missing-publication-candidate-race-skill" in str(exec_errors[0])
            assert not proposal_errors
            assert len(candidate_ids) == 1
            candidate_id = candidate_ids[0]
            candidate = runtime.store.get_tool_candidate(candidate_id)
            assert candidate is not None
            assert candidate.status.value == "proposed"
            descriptors = [
                obj
                for obj in runtime.store.list_objects_owned_by("process", pid)
                if obj.type == ObjectType.TOOL_CANDIDATE
                and isinstance(obj.payload, dict)
                and obj.payload.get("candidate_id") == candidate_id
            ]
            assert len(descriptors) == 1
            descriptor_oid = descriptors[0].oid
            publication = [
                item
                for item in runtime.store.list_runtime_publications(pid=pid)
                if item["kind"] == "process_exec"
            ][-1]
            assert publication["state"] == "rolled_back"
            assert not any(
                artifact.get("candidate_id") == candidate_id
                for artifact in publication["receipt"]["artifacts"]
            )
        finally:
            release_exec.set()
            runtime.close()

        for _attempt in range(2):
            reopened = Runtime.open(target, config=config)
            try:
                candidate = reopened.store.get_tool_candidate(candidate_id)
                assert candidate is not None
                assert candidate.status.value == "proposed"
                descriptor_rows = reopened.store.select_table_rows(
                    "objects",
                    "oid = ?",
                    (descriptor_oid,),
                )
                assert len(descriptor_rows) == 1
                assert descriptor_rows[0]["type"] == ObjectType.TOOL_CANDIDATE.value
                # Object payloads are deliberately volatile.  Reopen releases
                # the descriptor payload, but must not erase the exact Object
                # row or the durable candidate that the failed exec never owned.
                assert descriptor_rows[0]["lifecycle_state"] == "released"
                assert reopened.store.is_recovered_object_payload(descriptor_oid)
            finally:
                reopened.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_compensation_marker_prevents_reopen_from_replaying_over_later_candidate(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        candidate_id = ""
        descriptor_oid = ""
        publication_id = ""
        operation_id = ""
        try:
            runtime.tools.sandbox = _AcceptingValidationSandbox()
            pid = runtime.process.spawn(
                goal="preserve candidate proposed after exec compensation",
            )

            def fail_late_exec(*_args: object, **_kwargs: object) -> None:
                raise RuntimeError("injected late exec failure before terminal publication")

            original_update_operation = runtime.store.update_operation

            def fail_rolled_back_operation_update(
                record: Any,
                *,
                expected_states: Iterable[str] | None = None,
            ) -> bool:
                metadata = getattr(record, "metadata", {})
                if metadata.get("runtime_publication_state") == "rolled_back":
                    raise RuntimeError("injected rolled-back operation sink failure")
                return original_update_operation(
                    record,
                    expected_states=expected_states,
                )

            monkeypatch.setattr(
                runtime.image_boot,
                "_configure_skills",
                fail_late_exec,
            )
            monkeypatch.setattr(
                runtime.store,
                "update_operation",
                fail_rolled_back_operation_update,
            )

            with pytest.raises(RuntimePublicationPending) as caught:
                runtime.exec_process(
                    pid,
                    "base-agent:v0",
                    goal="terminal sink must remain recoverable",
                )

            publications = [
                item
                for item in runtime.store.list_runtime_publications(pid=pid)
                if item["kind"] == "process_exec"
            ]
            assert len(publications) == 1
            publication = publications[0]
            publication_id = str(publication["publication_id"])
            operation_id = str(publication["plan"].get("operation_id") or "")
            assert caught.value.publication_id == publication_id
            assert caught.value.operation_id == operation_id
            assert publication["state"] == "rollback_pending"
            assert publication["phase"] == "compensation_applied"
            applied_markers = [
                phase
                for phase in publication["receipt"]["phases"]
                if phase.get("phase") == "compensation_applied"
            ]
            assert applied_markers == [
                {"phase": "compensation_applied", "pid": pid}
            ]

            candidate_id = runtime.tools.propose(
                pid,
                {
                    "name": "candidate_after_applied_compensation",
                    "description": "Must survive terminalization after reopen.",
                    "input_schema": {"type": "object"},
                    "output_schema": {"type": "object"},
                },
                (
                    "export async function run(args: unknown, libos: unknown) "
                    "{ return { ok: true }; }\n"
                ),
            )
            candidate = runtime.store.get_tool_candidate(candidate_id)
            assert candidate is not None
            descriptors = [
                obj
                for obj in runtime.store.list_objects_owned_by("process", pid)
                if obj.type == ObjectType.TOOL_CANDIDATE
                and isinstance(obj.payload, dict)
                and obj.payload.get("candidate_id") == candidate_id
            ]
            assert len(descriptors) == 1
            descriptor_oid = descriptors[0].oid
            assert not any(
                artifact.get("candidate_id") == candidate_id
                for artifact in publication["receipt"]["artifacts"]
            )
        finally:
            runtime.close()

        first_terminal_publication: dict[str, object] | None = None
        for _attempt in range(2):
            reopened = Runtime.open(target, config=config)
            try:
                publication = reopened.store.get_runtime_publication(publication_id)
                assert publication is not None
                assert publication["state"] == "rolled_back"
                assert publication["phase"] == "startup_compensation_finalized"
                assert len(
                    [
                        phase
                        for phase in publication["receipt"]["phases"]
                        if phase.get("phase") == "compensation_applied"
                    ]
                ) == 1
                assert len(
                    [
                        phase
                        for phase in publication["receipt"]["phases"]
                        if phase.get("phase") == "recovery_claimed"
                    ]
                ) == 1
                if first_terminal_publication is None:
                    first_terminal_publication = publication
                else:
                    assert publication == first_terminal_publication

                candidate = reopened.store.get_tool_candidate(candidate_id)
                assert candidate is not None
                assert candidate.status.value == "proposed"
                descriptor_rows = reopened.store.select_table_rows(
                    "objects",
                    "oid = ?",
                    (descriptor_oid,),
                )
                assert len(descriptor_rows) == 1
                assert descriptor_rows[0]["type"] == ObjectType.TOOL_CANDIDATE.value
                assert descriptor_rows[0]["lifecycle_state"] == "released"
                assert reopened.store.is_recovered_object_payload(descriptor_oid)

                operation = reopened.store.get_operation(operation_id)
                assert operation is not None
                assert operation.state.value == "terminal"
                assert operation.outcome.value == "failed"
                assert operation.metadata["runtime_publication_id"] == publication_id
            finally:
                reopened.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
@pytest.mark.parametrize("entrypoint", ["runtime-wrapper", "image-boot-direct"])
def test_failed_compensation_fences_all_mutations_until_reopen_recovery(
    kind: str,
    entrypoint: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        publication_id = ""
        pid = ""
        try:
            runtime.tools.sandbox = _AcceptingValidationSandbox()
            pid = runtime.process.spawn(goal="recovery fence owner")
            finite = runtime.capability.issue_trusted(
                pid,
                "custom:recovery-fence-finite",
                [CapabilityRight.READ],
                issued_by="test",
                uses_remaining=8,
            )
            commit_reservation = runtime.capability.reserve_use(
                finite.cap_id,
                reserved_by="test",
                reason="recovery fence commit fixture",
            )
            restore_reservation = runtime.capability.reserve_use(
                finite.cap_id,
                reserved_by="test",
                reason="recovery fence restore fixture",
            )
            staged = runtime.capability.issue_trusted(
                pid,
                "custom:recovery-fence-staged",
                [CapabilityRight.READ],
                issued_by="test",
            )
            finalized = runtime.capability.issue_trusted(
                pid,
                "custom:recovery-fence-finalized",
                [CapabilityRight.READ],
                issued_by="test",
            )
            runtime.capability.stage_exec_revocation(
                finalized.cap_id,
                rollback_token="recovery-fence-finalize-token",
            )
            disabled = runtime.capability.issue_trusted(
                pid,
                "custom:recovery-fence-disabled",
                [CapabilityRight.READ],
                issued_by="test",
            )
            resource_revoked = runtime.capability.issue_trusted(
                pid,
                "custom:recovery-fence-resource",
                [CapabilityRight.READ],
                issued_by="test",
            )
            authority_decision = runtime.capability.authorize(
                pid,
                finite.resource,
                CapabilityRight.READ,
            )
            deferred_authority_transaction = runtime.capability.authority_transaction(
                [authority_decision],
                actor=pid,
                operation="recovery fence deferred transaction",
            )
            direct_draft = CapabilityDraft(
                subject=pid,
                resource="custom:recovery-fence-direct-draft",
                rights={CapabilityRight.READ.value},
                effect=CapabilityEffect.ALLOW,
                constraints={},
                metadata={},
                issued_by="test",
                issuer_cap_id=None,
                parent_cap_id=None,
                delegation_depth=0,
                max_delegation_depth=None,
                expires_at=None,
                uses_remaining=None,
                delegable=False,
                revocable=True,
            )

            def fail_late_exec(*_args: object, **_kwargs: object) -> None:
                raise RuntimeError("injected late exec failure for recovery fence")

            def fail_online_restore(*_args: object, **_kwargs: object) -> None:
                raise RuntimeError("injected online compensation restore failure")

            monkeypatch.setattr(
                runtime.image_boot,
                "_configure_skills",
                fail_late_exec,
            )
            monkeypatch.setattr(
                runtime.process_exec_state,
                "restore",
                fail_online_restore,
            )

            with pytest.raises(RuntimeRecoveryRequired) as caught:
                if entrypoint == "runtime-wrapper":
                    runtime.exec_process(
                        pid,
                        "base-agent:v0",
                        goal="must require reopen before more writes",
                    )
                else:
                    runtime.image_boot.exec(
                        pid,
                        "base-agent:v0",
                        goal="must require reopen before more writes",
                    )

            publication_id = caught.value.publication_id
            publication = runtime.store.get_runtime_publication(publication_id)
            assert publication is not None
            assert publication["state"] == "failed"
            assert publication["phase"] == "compensation_failed"
            assert not any(
                phase.get("phase") == "compensation_applied"
                for phase in publication["receipt"]["phases"]
            )
            assert runtime.lifecycle.state == "close_failed"
            assert runtime.lifecycle.shutdown_reason == (
                f"runtime.recovery_required:{publication_id}"
            )

            tables = (
                "audit_records",
                "capabilities",
                "capability_use_reservations",
                "events",
                "object_links",
                "object_namespaces",
                "objects",
                "operation_evidence",
                "operations",
                "processes",
                "runtime_publications",
                "skills",
                "tool_candidates",
                "tools",
            )
            before_rows = {
                table: deepcopy(runtime.store.select_table_rows(table))
                for table in tables
            }
            before_loaded_tool_ids = runtime.tools.loaded_tool_ids()

            # The read branch of mixed authorize APIs remains available for
            # diagnosis while their audit/evidence branch is mutation-fenced.
            runtime.capability.authorize(
                pid,
                finite.resource,
                CapabilityRight.READ,
            )

            def enter_deferred_authority_transaction() -> None:
                with deferred_authority_transaction:
                    pass

            rejected_capability_mutations = {
                "consume": lambda: runtime.capability.consume_use(
                    finite.cap_id,
                    used_by="rejected",
                ),
                "reserve": lambda: runtime.capability.reserve_use(
                    finite.cap_id,
                    reserved_by="rejected",
                ),
                "commit": lambda: runtime.capability.commit_reserved_use(
                    commit_reservation,
                    committed_by="rejected",
                    reason="must be fenced",
                ),
                "restore": lambda: runtime.capability.restore_reserved_use(
                    restore_reservation,
                    restored_by="rejected",
                ),
                "stage": lambda: runtime.capability.stage_exec_revocation(
                    staged.cap_id,
                    rollback_token="rejected-stage-token",
                ),
                "finalize": lambda: runtime.capability.finalize_exec_revocations(
                    pid,
                    rollback_token="recovery-fence-finalize-token",
                ),
                "disable": lambda: runtime.capability.disable_subject_capability(
                    disabled.cap_id,
                    actor="rejected",
                ),
                "revoke_resource": lambda: runtime.capability.revoke_resource_trusted(
                    resource_revoked.resource,
                    revoked_by="rejected",
                ),
                "require": lambda: runtime.capability.require(
                    pid,
                    finite.resource,
                    CapabilityRight.READ,
                    consume=False,
                ),
                "authority_transaction": enter_deferred_authority_transaction,
                "authorize_audit": lambda: runtime.capability.authorize(
                    pid,
                    finite.resource,
                    CapabilityRight.READ,
                    audit=True,
                ),
                "authorize_matching_audit": lambda: runtime.capability.authorize_matching_capabilities(
                    pid,
                    finite.resource,
                    CapabilityRight.READ,
                    [finite],
                    audit=True,
                ),
                "decision_from_matches_audit": lambda: runtime.capability.decision_from_matches(
                    subject=pid,
                    resource=finite.resource,
                    requested_right=CapabilityRight.READ.value,
                    matches=[finite],
                    selected_context={},
                    audit=True,
                ),
                "reauthorize_audit": lambda: runtime.capability.reauthorize_decision(
                    authority_decision,
                    audit=True,
                ),
                "transition_evidence": lambda: runtime.capability.transition_allowed_rights(
                    finite,
                    transition_kind="recovery-fence",
                    duplicates_authority=False,
                ),
                "direct_lease_consume": lambda: runtime.capability.leases.consume(
                    finite.cap_id,
                    used_by="rejected",
                ),
                "direct_lease_reserve": lambda: runtime.capability.leases.reserve(
                    finite.cap_id,
                    reserved_by="rejected",
                ),
                "direct_lease_reserve_decision": lambda: runtime.capability.leases.reserve_decision(
                    authority_decision,
                    used_by="rejected",
                    reason="must be fenced",
                ),
                "direct_lease_commit": lambda: runtime.capability.leases.commit(
                    commit_reservation,
                    committed_by="rejected",
                    reason="must be fenced",
                ),
                "direct_lease_restore": lambda: runtime.capability.leases.restore(
                    restore_reservation,
                    restored_by="rejected",
                ),
                "direct_mutation_issue": lambda: runtime.capability.mutations.issue(
                    direct_draft,
                    actor="rejected",
                    authority_decision=None,
                    attach_to_process=runtime.capability._attach_to_process,
                ),
                "direct_mutation_publish": lambda: runtime.capability.mutations.publish(
                    direct_draft,
                    attach_to_process=runtime.capability._attach_to_process,
                ),
                "direct_mutation_delegation_evidence": lambda: runtime.capability.mutations.record_delegation(
                    finite,
                    parent_cap=finite,
                    parent_subject=pid,
                    child_subject=pid,
                    actor="rejected",
                ),
                "direct_mutation_revoke": lambda: runtime.capability.mutations.revoke(
                    staged.cap_id,
                    revoked_by="rejected",
                    reason="must be fenced",
                    authority_decision=None,
                ),
                "direct_mutation_stage": lambda: runtime.capability.mutations.stage_exec_revocation(
                    staged.cap_id,
                    rollback_token="rejected-direct-stage-token",
                ),
                "direct_mutation_finalize": lambda: runtime.capability.mutations.finalize_exec_revocations(
                    pid,
                    rollback_token="recovery-fence-finalize-token",
                ),
                "direct_mutation_disable": lambda: runtime.capability.mutations.disable(
                    disabled.cap_id,
                    actor="rejected",
                ),
                "direct_mutation_revoke_resource": lambda: runtime.capability.mutations.revoke_resource(
                    resource_revoked.resource,
                    revoked_by="rejected",
                ),
            }
            for _name, mutate in rejected_capability_mutations.items():
                with pytest.raises(
                    RuntimeError,
                    match="not accepting operations: state=close_failed",
                ):
                    mutate()

            rejected_mutations = (
                lambda: runtime.tools.propose(
                    pid,
                    {
                        "name": "rejected_during_recovery_fence",
                        "description": "Must not be persisted.",
                        "input_schema": {"type": "object"},
                        "output_schema": {"type": "object"},
                    },
                    "export function run() { return {}; }",
                ),
                lambda: runtime.skills.activate_skill(
                    pid,
                    "missing-recovery-fence-skill",
                    require_capability=False,
                ),
                lambda: runtime.memory.create_object(
                    pid,
                    "artifact",
                    {"rejected": True},
                ),
                lambda: runtime.process.spawn(goal="rejected recovery-fence spawn"),
                lambda: runtime.capability.issue(
                    pid,
                    pid,
                    {
                        "resource": "custom:rejected-recovery-fence",
                        "rights": [CapabilityRight.READ.value],
                    },
                    require_authority=False,
                ),
            )
            for mutate in rejected_mutations:
                with pytest.raises(
                    RuntimeError,
                    match="not accepting operations: state=close_failed",
                ):
                    mutate()

            assert {
                table: runtime.store.select_table_rows(table)
                for table in tables
            } == before_rows
            assert runtime.tools.loaded_tool_ids() == before_loaded_tool_ids
        finally:
            _release_fenced_runtime_or_close(runtime)

        reopened = Runtime.open(target, config=config)
        candidate_id = ""
        descriptor_oid = ""
        try:
            assert reopened.lifecycle.state == "open"
            publication = reopened.store.get_runtime_publication(publication_id)
            assert publication is not None
            assert publication["state"] == "rolled_back"
            assert publication["phase"] == "startup_compensated"
            reopened.tools.sandbox = _AcceptingValidationSandbox()
            candidate_id = reopened.tools.propose(
                pid,
                {
                    "name": "allowed_after_reopen_recovery",
                    "description": "Must persist after recovery.",
                    "input_schema": {"type": "object"},
                    "output_schema": {"type": "object"},
                },
                "export function run() { return { recovered: true }; }",
            )
            descriptors = [
                obj
                for obj in reopened.store.list_objects_owned_by("process", pid)
                if obj.type == ObjectType.TOOL_CANDIDATE
                and isinstance(obj.payload, dict)
                and obj.payload.get("candidate_id") == candidate_id
            ]
            assert len(descriptors) == 1
            descriptor_oid = descriptors[0].oid
        finally:
            reopened.close()

        durable = Runtime.open(target, config=config)
        try:
            candidate = durable.store.get_tool_candidate(candidate_id)
            assert candidate is not None
            assert candidate.status.value == "proposed"
            descriptor_rows = durable.store.select_table_rows(
                "objects",
                "oid = ?",
                (descriptor_oid,),
            )
            assert len(descriptor_rows) == 1
            assert descriptor_rows[0]["type"] == ObjectType.TOOL_CANDIDATE.value
        finally:
            durable.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_recovery_fence_revokes_candidate_admitted_before_registry_barrier(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        pid = ""
        release_exec = threading.Event()
        try:
            runtime.tools.sandbox = _AcceptingValidationSandbox()
            pid = runtime.process.spawn(goal="revoke stale candidate admission")
            exec_paused = threading.Event()
            proposal_reached_barrier = threading.Event()
            proposal_finished = threading.Event()
            exec_errors: list[BaseException] = []
            proposal_errors: list[BaseException] = []
            candidate_ids: list[str] = []
            original_registry_lifecycle_lock = (
                runtime.tools._registry_lifecycle_lock
            )
            proposal_thread_name = "pre-fence-candidate-proposal"

            def observed_registry_lifecycle_lock() -> object:
                if threading.current_thread().name == proposal_thread_name:
                    proposal_reached_barrier.set()
                return original_registry_lifecycle_lock()

            def pause_then_fail(*_args: object, **_kwargs: object) -> None:
                exec_paused.set()
                if not release_exec.wait(timeout=10):
                    raise AssertionError("timed out waiting to fail exec")
                raise RuntimeError("injected late exec failure after admission")

            def fail_online_restore(*_args: object, **_kwargs: object) -> None:
                raise RuntimeError("injected compensation restore failure")

            def execute() -> None:
                try:
                    runtime.exec_process(pid, "base-agent:v0")
                except BaseException as exc:
                    exec_errors.append(exc)

            def propose() -> None:
                try:
                    candidate_ids.append(
                        runtime.tools.propose(
                            pid,
                            {
                                "name": "must_be_revoked_at_registry_barrier",
                                "description": "Must not cross a recovery fence.",
                                "input_schema": {"type": "object"},
                                "output_schema": {"type": "object"},
                            },
                            "export function run() { return {}; }",
                        )
                    )
                except BaseException as exc:
                    proposal_errors.append(exc)
                finally:
                    proposal_finished.set()

            monkeypatch.setattr(
                runtime.image_boot,
                "_configure_skills",
                pause_then_fail,
            )
            monkeypatch.setattr(
                runtime.process_exec_state,
                "restore",
                fail_online_restore,
            )
            monkeypatch.setattr(
                runtime.tools,
                "_registry_lifecycle_lock",
                observed_registry_lifecycle_lock,
            )

            exec_thread = threading.Thread(target=execute)
            proposal_thread = threading.Thread(
                target=propose,
                name=proposal_thread_name,
            )
            exec_thread.start()
            assert exec_paused.wait(timeout=10)
            proposal_thread.start()
            assert proposal_reached_barrier.wait(timeout=10)
            assert not proposal_finished.is_set()
            release_exec.set()
            exec_thread.join(timeout=15)
            proposal_thread.join(timeout=15)

            assert not exec_thread.is_alive()
            assert not proposal_thread.is_alive()
            assert len(exec_errors) == 1
            assert isinstance(exec_errors[0], RuntimeRecoveryRequired)
            assert candidate_ids == []
            assert len(proposal_errors) == 1
            assert isinstance(proposal_errors[0], RuntimeError)
            assert "state=close_failed" in str(proposal_errors[0])
            assert runtime.lifecycle.state == "close_failed"
            assert runtime.store.select_table_rows("tool_candidates") == []
        finally:
            release_exec.set()
            _release_fenced_runtime_or_close(runtime)

        reopened = Runtime.open(target, config=config)
        try:
            assert reopened.lifecycle.state == "open"
            assert reopened.store.select_table_rows("tool_candidates") == []
            assert not [
                obj
                for obj in reopened.store.list_objects_owned_by("process", pid)
                if obj.type == ObjectType.TOOL_CANDIDATE
            ]
        finally:
            reopened.close()


def test_forged_recovery_required_signal_does_not_poison_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="reject forged recovery signal")
        forged = RuntimeRecoveryRequired(
            publication_id="publication-forged",
            operation_id="operation-forged",
            pid=pid,
            state="failed",
            phase="compensation_failed",
        )

        def raise_forged(*_args: object, **_kwargs: object) -> None:
            raise forged

        monkeypatch.setattr(runtime.image_boot, "exec", raise_forged)
        with pytest.raises(RuntimeRecoveryRequired) as caught:
            runtime.exec_process(pid, "base-agent:v0")
        assert caught.value is forged
        assert runtime.lifecycle.state == "open"
        assert runtime.process.spawn(goal="allowed after forged signal")
    finally:
        runtime.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
@pytest.mark.parametrize("mutation", ["consume", "reserve"])
def test_capability_stale_store_waiter_revalidates_after_recovery_fence(
    kind: str,
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            pid = runtime.process.spawn(goal=f"stale capability {mutation} waiter")
            cap = runtime.capability.issue_trusted(
                pid,
                f"custom:stale-capability-{mutation}",
                [CapabilityRight.READ],
                issued_by="test",
                uses_remaining=2,
            )
            tables = (
                "audit_records",
                "capabilities",
                "capability_use_reservations",
                "events",
                "operation_evidence",
                "operations",
                "processes",
            )
            before = {
                table: deepcopy(runtime.store.select_table_rows(table))
                for table in tables
            }

            repository = runtime.capability.leases.store
            original_transaction = repository.transaction
            waiter_thread_id: int | None = None
            reached_store_barrier = threading.Event()

            def instrumented_transaction(
                *,
                include_object_payloads: bool = False,
            ) -> contextlib.AbstractContextManager[object]:
                if threading.get_ident() == waiter_thread_id:
                    reached_store_barrier.set()
                return original_transaction(
                    include_object_payloads=include_object_payloads,
                )

            monkeypatch.setattr(repository, "transaction", instrumented_transaction)
            errors: list[BaseException] = []

            def mutate_after_admission() -> None:
                nonlocal waiter_thread_id
                waiter_thread_id = threading.get_ident()
                try:
                    with runtime.lifecycle.admit():
                        if mutation == "consume":
                            runtime.capability.leases.consume(
                                cap.cap_id,
                                used_by="stale-waiter",
                            )
                        else:
                            runtime.capability.leases.reserve(
                                cap.cap_id,
                                reserved_by="stale-waiter",
                            )
                except BaseException as exc:
                    errors.append(exc)

            # Hold the real backend lock without opening an unadmitted durable
            # transaction.  The waiter still blocks at its transaction
            # boundary, but fencing recovery cannot turn the barrier itself
            # into a commit that must be rejected.
            with runtime.store.locked():
                waiter = threading.Thread(target=mutate_after_admission)
                waiter.start()
                assert reached_store_barrier.wait(timeout=5)
                with runtime.lifecycle.admit():
                    runtime.lifecycle.mark_recovery_required(
                        publication_id=f"stale-capability-{mutation}",
                    )

            waiter.join(timeout=10)
            assert not waiter.is_alive()
            assert len(errors) == 1
            assert isinstance(errors[0], RuntimeError)
            assert "not accepting operations: state=close_failed" in str(errors[0])
            assert {
                table: runtime.store.select_table_rows(table)
                for table in tables
            } == before
        finally:
            _release_fenced_runtime_or_close(runtime)

        reopened = Runtime.open(target, config=config)
        try:
            assert reopened.lifecycle.state == "open"
            reopened.capability.issue_trusted(
                "recovered-capability-owner",
                f"custom:recovered-capability-{mutation}",
                [CapabilityRight.READ],
                issued_by="test",
            )
        finally:
            reopened.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_authority_transaction_rolls_back_business_and_settlement_after_recovery_fence(
    kind: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            pid = runtime.process.spawn(goal="stale authority transaction")
            cap = runtime.capability.issue_trusted(
                pid,
                "custom:stale-authority-transaction",
                [CapabilityRight.READ],
                issued_by="test",
                uses_remaining=1,
            )
            decision = runtime.capability.authorize(
                pid,
                cap.resource,
                CapabilityRight.READ,
            )
            deferred = runtime.capability.authority_transaction(
                [decision],
                actor=pid,
                operation="stale authority transaction",
            )
            tables = (
                "audit_records",
                "capabilities",
                "capability_use_reservations",
                "events",
                "operation_evidence",
                "operations",
                "processes",
            )
            before = {
                table: deepcopy(runtime.store.select_table_rows(table))
                for table in tables
            }
            business_applied = threading.Event()
            release_business = threading.Event()
            errors: list[BaseException] = []

            def transact() -> None:
                try:
                    with deferred:
                        runtime.capability.mutations.disable(
                            cap.cap_id,
                            actor="stale-authority-transaction",
                        )
                        business_applied.set()
                        assert release_business.wait(timeout=10)
                except BaseException as exc:
                    errors.append(exc)

            worker = threading.Thread(target=transact)
            worker.start()
            assert business_applied.wait(timeout=10)
            with runtime.lifecycle.admit():
                runtime.lifecycle.mark_recovery_required(
                    publication_id="stale-authority-transaction",
                )
            release_business.set()
            worker.join(timeout=15)

            assert not worker.is_alive()
            assert len(errors) == 1
            assert isinstance(errors[0], RuntimeError)
            assert "not accepting operations: state=close_failed" in str(errors[0])
            assert {
                table: runtime.store.select_table_rows(table)
                for table in tables
            } == before
            restored = runtime.store.get_capability(cap.cap_id)
            assert restored is not None
            assert restored.active
            assert restored.uses_remaining == 1
        finally:
            _release_fenced_runtime_or_close(runtime)

        reopened = Runtime.open(target, config=config)
        try:
            assert reopened.lifecycle.state == "open"
            restored = reopened.store.get_capability(cap.cap_id)
            assert restored is not None
            assert restored.active
            assert restored.uses_remaining == 1
        finally:
            reopened.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_authority_outer_commit_is_atomic_with_recovery_fence(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fence that wins immediately before commit rolls back the whole UoW."""

    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        release_commit = threading.Event()
        worker: threading.Thread | None = None
        fencer: threading.Thread | None = None
        original_guard = runtime.store._admission_commit_guard
        assert original_guard is not None
        try:
            pid = runtime.process.spawn(goal="atomic admission commit fence")
            authority = runtime.capability.issue_trusted(
                pid,
                "custom:atomic-admission-commit-authority",
                [CapabilityRight.READ],
                issued_by="test",
                uses_remaining=1,
            )
            target_capability = runtime.capability.issue_trusted(
                pid,
                "custom:atomic-admission-commit-target",
                [CapabilityRight.READ],
                issued_by="test",
            )
            decision = runtime.capability.authorize(
                pid,
                authority.resource,
                CapabilityRight.READ,
            )
            tables = (
                "audit_records",
                "capabilities",
                "capability_use_reservations",
                "events",
                "operation_evidence",
                "operations",
                "processes",
            )
            before = {
                table: deepcopy(runtime.store.select_table_rows(table))
                for table in tables
            }

            arm_commit_barrier = threading.Event()
            commit_waiting = threading.Event()
            fence_finished = threading.Event()
            worker_thread_id: int | None = None

            @contextlib.contextmanager
            def blocked_commit_guard() -> Iterator[None]:
                if (
                    arm_commit_barrier.is_set()
                    and threading.get_ident() == worker_thread_id
                ):
                    commit_waiting.set()
                    assert release_commit.wait(timeout=10)
                with original_guard():
                    yield

            monkeypatch.setattr(
                runtime.store,
                "_admission_commit_guard",
                blocked_commit_guard,
            )
            worker_errors: list[BaseException] = []
            fence_errors: list[BaseException] = []

            def transact() -> None:
                nonlocal worker_thread_id
                worker_thread_id = threading.get_ident()
                try:
                    with runtime.capability.authority_transaction(
                        [decision],
                        actor=pid,
                        operation="atomic admission commit fence",
                    ):
                        runtime.capability.mutations.disable(
                            target_capability.cap_id,
                            actor="atomic-admission-commit-fence",
                        )
                        arm_commit_barrier.set()
                except BaseException as exc:
                    worker_errors.append(exc)

            def fence() -> None:
                try:
                    with runtime.lifecycle.admit():
                        runtime.lifecycle.mark_recovery_required(
                            publication_id="atomic-admission-commit-fence",
                        )
                except BaseException as exc:
                    fence_errors.append(exc)
                finally:
                    fence_finished.set()

            worker = threading.Thread(target=transact)
            worker.start()
            assert commit_waiting.wait(timeout=10)

            fencer = threading.Thread(target=fence)
            fencer.start()
            # The fencer never waits on the store lock. This both fixes the race
            # and ratchets the required store -> lifecycle lock order.
            assert fence_finished.wait(timeout=10)
            assert fence_errors == []

            release_commit.set()
            worker.join(timeout=15)
            fencer.join(timeout=15)
            assert not worker.is_alive()
            assert not fencer.is_alive()
            assert len(worker_errors) == 1
            assert isinstance(worker_errors[0], RuntimeError)
            assert "not accepting operations: state=close_failed" in str(
                worker_errors[0]
            )
            assert {
                table: runtime.store.select_table_rows(table)
                for table in tables
            } == before
            restored_authority = runtime.store.get_capability(authority.cap_id)
            restored_target = runtime.store.get_capability(target_capability.cap_id)
            assert restored_authority is not None
            assert restored_authority.active
            assert restored_authority.uses_remaining == 1
            assert restored_target is not None
            assert restored_target.active
        finally:
            release_commit.set()
            if worker is not None and worker.is_alive():
                worker.join(timeout=15)
            if fencer is not None and fencer.is_alive():
                fencer.join(timeout=15)
            monkeypatch.setattr(
                runtime.store,
                "_admission_commit_guard",
                original_guard,
            )
            _release_fenced_runtime_or_close(runtime)

        reopened = Runtime.open(target, config=config)
        try:
            restored_authority = reopened.store.get_capability(authority.cap_id)
            restored_target = reopened.store.get_capability(target_capability.cap_id)
            assert restored_authority is not None
            assert restored_authority.active
            assert restored_authority.uses_remaining == 1
            assert restored_target is not None
            assert restored_target.active
        finally:
            reopened.close()


def test_recovery_signal_association_failure_still_poisons_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="poison before association error")

        def fail_late_exec(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("injected late exec failure")

        def fail_online_restore(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("injected compensation restore failure")

        original_get_operation = runtime.store.get_operation
        association_failure_injected = False

        def fail_first_fenced_operation_read(operation_id: str) -> object:
            nonlocal association_failure_injected
            if (
                runtime.lifecycle.state == "close_failed"
                and not association_failure_injected
            ):
                association_failure_injected = True
                return None
            return original_get_operation(operation_id)

        monkeypatch.setattr(
            runtime.image_boot,
            "_configure_skills",
            fail_late_exec,
        )
        monkeypatch.setattr(
            runtime.process_exec_state,
            "restore",
            fail_online_restore,
        )
        monkeypatch.setattr(
            runtime.store,
            "get_operation",
            fail_first_fenced_operation_read,
        )

        with pytest.raises(
            ValidationError,
            match="recovery signal operation binding is invalid",
        ):
            runtime.image_boot.exec(pid, "base-agent:v0")
        assert association_failure_injected
        assert runtime.lifecycle.state == "close_failed"
    finally:
        _release_fenced_runtime_or_close(runtime)


def test_missing_operation_binding_cannot_bypass_recovery_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="poison before missing binding error")

        def damage_binding_then_fail(
            *_args: object,
            **_kwargs: object,
        ) -> None:
            publication_id = str(_kwargs["publication_id"])
            publication = runtime.store.get_runtime_publication(publication_id)
            assert publication is not None
            damaged_plan = deepcopy(publication["plan"])
            damaged_plan["operation_id"] = None
            with runtime.store.transaction() as cursor:
                updated = cursor.execute(
                    "UPDATE runtime_publications SET plan_json = ? "
                    "WHERE publication_id = ?",
                    (dumps(damaged_plan), publication_id),
                )
                assert updated.rowcount == 1
            raise RuntimeError("injected late exec failure after binding damage")

        def fail_online_restore(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("injected compensation restore failure")

        monkeypatch.setattr(
            runtime.image_boot,
            "_configure_skills",
            damage_binding_then_fail,
        )
        monkeypatch.setattr(
            runtime.process_exec_state,
            "restore",
            fail_online_restore,
        )

        with pytest.raises(
            ValidationError,
            match="recovery signal operation binding is invalid",
        ):
            runtime.image_boot.exec(pid, "base-agent:v0")

        publication = [
            item
            for item in runtime.store.list_runtime_publications(pid=pid)
            if item["kind"] == "process_exec"
        ][-1]
        assert publication["state"] == "rollback_pending"
        assert publication["phase"] == "compensating"
        assert publication["plan"]["operation_id"] is None
        assert not any(
            phase.get("phase") == "compensation_applied"
            for phase in publication["receipt"]["phases"]
        )
        assert runtime.lifecycle.state == "close_failed"
        assert runtime.lifecycle.shutdown_reason == (
            f"runtime.recovery_required:{publication['publication_id']}"
        )
    finally:
        _release_fenced_runtime_or_close(runtime)


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_unknown_receipt_handler_stays_failed_then_fails_reopen_closed(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _persistent_target(kind, tmp_path, recovery_max_attempts=1) as (
        target,
        config,
    ):
        runtime = Runtime.open(target, config=config)
        publication_id = f"publication-unknown-contract-{uuid4().hex}"
        try:
            pid = runtime.process.spawn(goal="unknown publication receipt contract")
            before = runtime.process_exec_state.capture(pid)
            runtime.store.insert_runtime_publication(
                publication_id=publication_id,
                kind="process_exec",
                pid=pid,
                owner_instance_id="runtime-that-crashed",
                plan={
                    "pid": pid,
                    "image_id": "base-agent:v0",
                    "before_snapshot": before.snapshot.to_mapping(),
                    "before_tool_ids": sorted(before.tool_ids),
                },
            )
            assert runtime.store.record_runtime_publication_artifact(
                publication_id,
                {
                    "artifact_id": "unknown:contract-effect",
                    "kind": "contract_effect_without_compensation_handler",
                },
            )
            assert runtime.store.advance_runtime_publication(
                publication_id,
                state="applying",
                phase="contract_effect_applied",
                expected_states={"planning"},
            )
            monkeypatch.setattr(
                runtime.image_boot,
                "_require_recovery_lease",
                lambda: None,
            )

            with pytest.raises(
                ValidationError,
                match="cannot recover process exec publication",
            ):
                runtime.image_boot.recover_incomplete_publications()
            failed = runtime.store.get_runtime_publication(publication_id)
            assert failed is not None
            assert failed["state"] == "failed"
            assert failed["receipt"]["recovery"]["attempt"] == 1

            with pytest.raises(ValidationError, match="requires manual recovery"):
                runtime.image_boot.recover_incomplete_publications()
            manual = runtime.store.get_runtime_publication(publication_id)
            assert manual is not None
            assert manual["state"] == "manual"
            assert manual["receipt"]["recovery"]["attempt"] == 2
            assert manual["receipt"]["recovery"]["disposition"] == "manual"
        finally:
            runtime.close()

        for _attempt in range(2):
            with pytest.raises(ValidationError, match="requires manual recovery"):
                Runtime.open(target, config=config)
        durable_store = open_store(target, config=config)
        try:
            durable = durable_store.get_runtime_publication(publication_id)
            assert durable is not None
            assert durable["state"] == "manual"
            assert durable["phase"] == "recovery_attempts_exhausted"
            assert durable["receipt"]["recovery"]["attempt"] == 2
        finally:
            durable_store.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_manual_launch_publication_fails_every_reopen_closed(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _persistent_target(kind, tmp_path, recovery_max_attempts=1) as (
        target,
        config,
    ):
        runtime = Runtime.open(target, config=config)
        publication_id = f"publication-manual-launch-{uuid4().hex}"
        try:
            runtime.store.insert_runtime_publication(
                publication_id=publication_id,
                kind="process_launch",
                pid=f"pid-manual-launch-{uuid4().hex}",
                owner_instance_id="runtime-that-crashed",
                plan={"image_id": "base-agent:v0"},
            )
            assert runtime.store.record_runtime_publication_artifact(
                publication_id,
                {
                    "artifact_id": "unknown:launch-contract-effect",
                    "kind": "launch_effect_without_compensation_handler",
                },
            )
            assert runtime.store.advance_runtime_publication(
                publication_id,
                state="applying",
                phase="launch_effect_applied",
                expected_states={"planning"},
            )
            monkeypatch.setattr(
                runtime.process,
                "_require_recovery_lease",
                lambda: None,
            )
            with pytest.raises(
                ProcessError,
                match="cannot compensate process publication",
            ):
                runtime.process.recover_incomplete_publications()
            failed = runtime.store.get_runtime_publication(publication_id)
            assert failed is not None
            assert failed["state"] == "failed"
            assert failed["receipt"]["recovery"]["attempt"] == 1
            with pytest.raises(ProcessError, match="requires manual recovery"):
                runtime.process.recover_incomplete_publications()
        finally:
            runtime.close()

        for _attempt in range(2):
            with pytest.raises(ProcessError, match="requires manual recovery"):
                Runtime.open(target, config=config)
        durable_store = open_store(target, config=config)
        try:
            durable = durable_store.get_runtime_publication(publication_id)
            assert durable is not None
            assert durable["state"] == "manual"
            assert durable["receipt"]["recovery"]["attempt"] == 2
            assert durable["receipt"]["recovery"]["disposition"] == "manual"
        finally:
            durable_store.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_reopen_takes_over_orphaned_claim_once_before_jit_rehydrate(
    kind: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        publication_id = f"publication-orphaned-claim-{uuid4().hex}"
        try:
            pid = runtime.process.spawn(goal="orphaned recovery claim contract")
            before = runtime.process_exec_state.capture(pid)
            process = runtime.process.get(pid)
            with runtime.store.transaction():
                admission_token = runtime.store.claim_host_process_exec(
                    pid,
                    owner_id="runtime-that-crashed:process.exec",
                    expected_revision=process.revision,
                    expected_state_generation=process.state_generation,
                    expected_execution_generation=process.execution_generation,
                )
                assert admission_token is not None
                publication = runtime.store.insert_runtime_publication(
                    publication_id=publication_id,
                    kind="process_exec",
                    pid=pid,
                    owner_instance_id="runtime-that-crashed",
                    plan={
                        "pid": pid,
                        "image_id": "base-agent:v0",
                        "before_snapshot": before.snapshot.to_mapping(),
                        "before_tool_ids": sorted(before.tool_ids),
                        "admission_execution_generation": (
                            admission_token.generation
                        ),
                        "admission_execution_owner_id": admission_token.owner_id,
                        "admission_execution_lease_id": admission_token.lease_id,
                    },
                )
            orphaned = runtime.store.claim_runtime_publication_recovery(
                publication_id,
                claimant_instance_id="recovery-that-crashed",
                expected_owner_instance_id=publication["owner_instance_id"],
                expected_state=publication["state"],
                classification="compensate_process_exec",
                max_attempts=3,
            )
            assert orphaned is not None
            orphaned_lease = orphaned["receipt"]["recovery"]["lease_id"]
        finally:
            runtime.close()

        reopened = Runtime.open(target, config=config)
        try:
            recovered = reopened.store.get_runtime_publication(publication_id)
            assert recovered is not None
            assert recovered["state"] == "rolled_back"
            assert recovered["phase"] == "startup_compensated"
            assert publication_id in reopened.recovered_exec_publications
            recovery = recovered["receipt"]["recovery"]
            assert recovery["attempt"] == 2
            assert recovery["disposition"] == "terminal"
            assert recovery["lease_id"] != orphaned_lease
        finally:
            reopened.close()

        second_reopen = Runtime.open(target, config=config)
        try:
            stable = second_reopen.store.get_runtime_publication(publication_id)
            assert stable == recovered
            assert publication_id not in second_reopen.recovered_exec_publications
        finally:
            second_reopen.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_pre_handler_recovery_failure_persists_attempt_then_manual(
    kind: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path, recovery_max_attempts=1) as (
        target,
        config,
    ):
        runtime = Runtime.open(target, config=config)
        publication_id = f"publication-invalid-plan-{uuid4().hex}"
        try:
            pid = runtime.process.spawn(goal="invalid recovery plan contract")
            runtime.store.insert_runtime_publication(
                publication_id=publication_id,
                kind="process_exec",
                pid=pid,
                owner_instance_id="runtime-that-crashed",
                plan={"pid": pid, "image_id": "base-agent:v0"},
            )
            assert runtime.store.advance_runtime_publication(
                publication_id,
                state="applying",
                phase="process_exec_applied",
                expected_states={"planning"},
            )
        finally:
            runtime.close()

        with pytest.raises(
            ValidationError,
            match="cannot recover process exec publication",
        ):
            Runtime.open(target, config=config)
        failed_store = open_store(target, config=config)
        try:
            failed = failed_store.get_runtime_publication(publication_id)
            assert failed is not None
            assert failed["state"] == "failed"
            assert failed["receipt"]["recovery"]["attempt"] == 1
        finally:
            failed_store.close()

        for _attempt in range(2):
            with pytest.raises(ValidationError, match="requires manual recovery"):
                Runtime.open(target, config=config)
        manual_store = open_store(target, config=config)
        try:
            manual = manual_store.get_runtime_publication(publication_id)
            assert manual is not None
            assert manual["state"] == "manual"
            assert manual["receipt"]["recovery"]["attempt"] == 2
        finally:
            manual_store.close()


def _assert_owned_artifacts_absent(
    runtime: Runtime,
    *,
    substrate_root: Path,
    capability_ids: set[str],
    tool_ids: set[str],
    candidate_ids: set[str],
    workspace_paths: set[str],
) -> None:
    assert all(
        runtime.store.get_capability(capability_id) is None
        for capability_id in capability_ids
    )
    remaining_tool_ids = {str(row["tool_id"]) for row in runtime.store.list_tools()}
    assert not (tool_ids & remaining_tool_ids)
    assert all(
        runtime.store.get_tool_candidate(candidate_id) is None
        for candidate_id in candidate_ids
    )
    assert all(
        not (substrate_root / workspace_path).exists()
        for workspace_path in workspace_paths
    )


def _write_image_package(root: Path) -> Path:
    root.mkdir(parents=True)
    root.joinpath("IMAGE.yaml").write_text(
        """
image_id: package-agent:v0
name: package-agent
version: v0
prompt: prompt.md
default_tools:
  - human_output
context_policy: evidence_first
safety_profile: publication-contract
jit_tools: tools/jit-tools.json
workspace:
  source: workspace
  working_directory: .
  grants:
    - path: .
      rights: [read, write]
      recursive: true
""".lstrip(),
        encoding="utf-8",
    )
    root.joinpath("prompt.md").write_text(
        "Publication rollback contract image.\n",
        encoding="utf-8",
    )
    workspace = root / "workspace"
    workspace.mkdir()
    workspace.joinpath("seed.txt").write_text("seed\n", encoding="utf-8")
    scripts = root / "tools" / "scripts"
    scripts.mkdir(parents=True)
    root.joinpath("tools", "jit-tools.json").write_text(
        """
[
  {
    "name": "publication_count",
    "description": "Count text characters.",
    "source_path": "tools/scripts/publication_count.ts",
    "input_schema": {"type": "object"},
    "output_schema": {"type": "object"},
    "tests": []
  }
]
""".strip(),
        encoding="utf-8",
    )
    scripts.joinpath("publication_count.ts").write_text(
        "export function run(args, libos) { "
        "return { count: String(args.text || '').length }; }\n",
        encoding="utf-8",
    )
    return root


@contextlib.contextmanager
def _persistent_target(
    kind: str,
    tmp_path: Path,
    *,
    recovery_max_attempts: int | None = None,
) -> Iterator[tuple[str | Path, AgentLibOSConfig]]:
    if kind == "sqlite-file":
        runtime_defaults = RuntimeDefaults(
            **(
                {"publication_recovery_max_attempts": recovery_max_attempts}
                if recovery_max_attempts is not None
                else {}
            )
        )
        yield tmp_path / "publication-contract.sqlite", AgentLibOSConfig(
            runtime=runtime_defaults
        )
        return
    if kind != "postgres":  # pragma: no cover - parametrization contract
        raise AssertionError(f"unknown backend: {kind}")

    dsn = os.environ["AGENT_LIBOS_POSTGRES_DSN"]
    schema = f"agent_libos_publication_contract_{uuid4().hex}"
    import psycopg
    from psycopg import sql

    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    target = _dsn_with_search_path(dsn, schema)
    try:
        yield target, AgentLibOSConfig(
            runtime=RuntimeDefaults(
                store_backend="postgres",
                store_dsn=target,
                **(
                    {"publication_recovery_max_attempts": recovery_max_attempts}
                    if recovery_max_attempts is not None
                    else {}
                ),
            )
        )
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )


def _dsn_with_search_path(dsn: str, schema: str) -> str:
    parsed = urlsplit(dsn)
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key != "options"
    ]
    query.append(("options", f"-csearch_path={schema}"))
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment)
    )
