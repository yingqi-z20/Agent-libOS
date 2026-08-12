from __future__ import annotations

import json
import base64
import hashlib
import hmac
import os
import subprocess
import time
import threading
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from agent_libos import Runtime
from agent_libos.capability.effect_binding import canonical_effect_hash
from agent_libos.config import AgentLibOSConfig, SemanticDefaults
from agent_libos.models import (
    CapabilityRight,
    DataFlowContext,
    DataFlowDirection,
    DataLabels,
    DataSink,
    DataSourceRef,
    EventType,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    GitDiffResult,
    GitPath,
    GitStateToken,
    GitStatusEntry,
    GitStatusKind,
    GitStatusResult,
    HumanRequest,
    HumanRequestStatus,
    JsonRpcCallResult,
    JsonRpcCallStatus,
    McpCallResult,
    McpCallStatus,
    ObjectMetadata,
    ObjectPatch,
    ObjectType,
    ProcessStatus,
    SinkTrustLevel,
    SinkTrustRule,
)
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.sdk import (
    ProtectedOperationContract,
    ProtectedOperationEvidence,
    ProtectedOperationInvocation,
    ProviderRegistryBinding,
    ProviderPhase,
    ResourcePolicy,
)
from agent_libos.semantic.external import ProtectedSemanticAssessmentCall
from agent_libos.semantic.protected import SdkProtectedSemanticCallPort
from agent_libos.primitives.git import _GitFlowSnapshot, _GitReadFlowSnapshot
from agent_libos.sdk.protected_operations import (
    _post_commit_result_identity,
    visit_bounded_host_result_text,
)
from agent_libos.substrate import (
    CommandMetrics,
    CommandResult,
    LocalResourceProviderSubstrate,
    PathState,
)
from agent_libos.utils.ids import utc_now
from agent_libos.utils.serde import to_jsonable
from tests.support.runtime import temporary_runtime


_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_HOST_RESULT_SENTINEL = "SEMANTIC_HOST_RESULT_SECRET_SENTINEL"


class _Provider:
    def classify_external_effect(self, _operation, _context, _result):
        return ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
            rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
            state_mutation=False,
            information_flow=True,
            metadata={},
        )


def _evidence(pid: str) -> ProtectedOperationEvidence:
    return ProtectedOperationEvidence(
        event_type=EventType.EXTERNAL_READ,
        event_source=pid,
        event_target="test:item",
        event_payload={"ok": True},
        audit_action="primitive.semantic-test.read",
        audit_actor=pid,
        audit_target="test:item",
        audit_decision={"ok": True},
    )


def _protected_fixture(
    runtime,
    *,
    data_flow_direction: DataFlowDirection = DataFlowDirection.NONE,
):
    pid = runtime.process.spawn(goal="semantic observer fixture")
    capability = runtime.capability.issue_trusted(
        pid,
        "test:item",
        [CapabilityRight.READ],
        issued_by="test",
    )
    decision = runtime.capability.require(
        pid,
        "test:item",
        CapabilityRight.READ,
        consume=False,
    )
    contract = ProtectedOperationContract(
        name="primitive.semantic-test.read",
        provider="test",
        operation="read",
        evidence_roles=("audit", "event", "effect"),
        resource_policy=ResourcePolicy.NONE,
        information_flow=True,
        data_flow_direction=data_flow_direction,
    )
    runtime.protected_operations.register_contract(contract)
    invocation = ProtectedOperationInvocation(
        pid=pid,
        actor=pid,
        target="test:item",
        decisions=(decision,),
        canonical_args={"item_sha256": _A},
        observation={"item_sha256": _A},
    )
    return pid, capability, contract, invocation


def test_process_post_commit_spawn_observer_is_late_and_failure_isolated() -> None:
    sentinel = "SEMANTIC_SPAWN_OBSERVER_SECRET_SENTINEL"
    with temporary_runtime() as runtime:
        observed: list[tuple[str, str]] = []

        def fail(_pid: str, _image: str, _publication_id: str) -> None:
            raise RuntimeError(sentinel)

        def inspect_committed(pid: str, _image: str, publication_id: str) -> None:
            process = runtime.store.get_process(pid)
            publication = runtime.store.get_runtime_publication(publication_id)
            assert process is not None and process.status is ProcessStatus.RUNNABLE
            assert publication is not None and publication["state"] == "committed"
            observed.append((pid, publication_id))

        runtime.process.add_post_commit_spawn_observer(fail)
        runtime.process.add_post_commit_spawn_observer(inspect_committed)
        pid = runtime.process.spawn(goal="observe only after commit")

        assert observed and observed[0][0] == pid
        failures = runtime.process.post_commit_spawn_observer_failures
        assert len(failures) == 1
        assert failures[0]["pid"] == pid
        assert failures[0]["error_type"] == "RuntimeError"
        assert sentinel not in repr(failures)


def test_sdk_default_and_invocation_observers_are_additive_and_isolated() -> None:
    with temporary_runtime() as runtime:
        pid, _capability, contract, invocation = _protected_fixture(runtime)
        default_results: list[tuple[object, object]] = []
        invocation_results: list[object] = []
        default_failures: list[object] = []
        invocation_failures: list[object] = []

        def observe_default(result, observation) -> None:
            default_results.append((result, observation))
            if result == "default-fails":
                raise RuntimeError("default observer failure is isolated")

        runtime.protected_operations.bind_post_commit_result_observer(
            observe_default,
            failure=default_failures.append,
        )
        with pytest.raises(RuntimeError, match="already bound"):
            runtime.protected_operations.bind_post_commit_result_observer(
                lambda _result, _observation: None
            )

        with runtime.protected_operations.start(
            contract,
            invocation,
            provider=_Provider(),
        ) as operation:
            result = operation.call(ProviderPhase("read", information_flow=True), lambda: "first")
            assert operation.complete(result, _evidence(pid)) == "first"
        assert len(default_results) == 1
        assert default_results[0][1].contract_name == contract.name
        assert default_results[0][1].result_sha256 == hashlib.sha256(
            b'"first"'
        ).hexdigest()
        assert default_results[0][1].result_descriptor["digest_mode"] == (
            "canonical_bounded"
        )

        additive = replace(
            invocation,
            post_commit_result_observer=(
                lambda result, _observation: invocation_results.append(result)
            ),
            post_commit_observer_failure=invocation_failures.append,
        )
        with runtime.protected_operations.start(
            contract,
            additive,
            provider=_Provider(),
        ) as operation:
            result = operation.call(
                ProviderPhase("read", information_flow=True),
                lambda: "default-fails",
            )
            assert operation.complete(result, _evidence(pid)) == "default-fails"
            assert operation.post_commit_observer_failure is not None

        def fail_invocation(result, _observation) -> None:
            invocation_results.append(result)
            raise RuntimeError("invocation observer failure is isolated")

        additive_failure = replace(
            invocation,
            post_commit_result_observer=fail_invocation,
            post_commit_observer_failure=invocation_failures.append,
        )
        with runtime.protected_operations.start(
            contract,
            additive_failure,
            provider=_Provider(),
        ) as operation:
            result = operation.call(
                ProviderPhase("read", information_flow=True),
                lambda: "invocation-fails",
            )
            assert operation.complete(result, _evidence(pid)) == "invocation-fails"
            assert operation.post_commit_observer_failure is not None

        assert invocation_results == ["default-fails", "invocation-fails"]
        assert [item[0] for item in default_results] == [
            "first",
            "default-fails",
            "invocation-fails",
        ]
        assert len(default_failures) == 1
        assert len(invocation_failures) == 1


def test_sdk_result_descriptor_is_bounded_and_carries_ingress_labels() -> None:
    metaclass_hook_calls = 0
    metaclass_comparison_calls = 0
    property_hook_calls = 0

    class HostileMeta(type):
        def __getattribute__(cls, name):
            nonlocal metaclass_hook_calls
            if name in {"__module__", "__qualname__", "__dataclass_fields__"}:
                metaclass_hook_calls += 1
                raise AssertionError("provider metaclass hooks must not run")
            return super().__getattribute__(name)

        def __hash__(cls):
            nonlocal metaclass_comparison_calls
            metaclass_comparison_calls += 1
            raise AssertionError("provider metaclass hash hooks must not run")

        def __eq__(cls, _other):
            nonlocal metaclass_comparison_calls
            metaclass_comparison_calls += 1
            raise AssertionError("provider metaclass equality hooks must not run")

    def hostile_to_dict(_self):
        raise AssertionError("arbitrary provider hooks must not run")

    HostileResult = HostileMeta(
        _HOST_RESULT_SENTINEL,
        (),
        {"to_dict": hostile_to_dict},
    )

    class SpoofedHostResult(metaclass=HostileMeta):
        __module__ = "agent_libos.primitives.filesystem"
        __qualname__ = "FileReadResult"

        @property
        def content(self):
            nonlocal property_hook_calls
            property_hook_calls += 1
            raise AssertionError("provider properties must not run")

    with temporary_runtime() as runtime:
        pid, _capability, contract, invocation = _protected_fixture(
            runtime,
            data_flow_direction=DataFlowDirection.INGRESS,
        )
        observations: list[object] = []
        labels = DataLabels(sensitivity="secret", integrity="untrusted")
        source = DataSourceRef(oid="obj-source", version=1, content_sha256=_A)
        invocation = replace(
            invocation,
            data_flow_ingress_context=DataFlowContext(
                labels=labels,
                source_refs=(source,),
            ),
        )
        runtime.protected_operations.bind_post_commit_result_observer(
            lambda _result, observation: observations.append(observation)
        )
        hostile = HostileResult()
        with runtime.protected_operations.start(
            contract,
            invocation,
            provider=_Provider(),
        ) as operation:
            result = operation.call(
                ProviderPhase("read", information_flow=True),
                lambda: hostile,
            )
            assert operation.complete(result, _evidence(pid)) is hostile
        cyclic: list[object] = []
        cyclic.append(cyclic)
        oversized = "x" * (256 * 1024 + 1)
        cumulative_oversized = ["z" * (32 * 1024) for _ in range(9)]
        colliding_keys = {1: "integer", "1": "string"}
        binary = b"\x00provider-bytes\xff"
        for selected in (
            cyclic,
            oversized,
            cumulative_oversized,
            colliding_keys,
            SpoofedHostResult(),
            binary,
        ):
            with runtime.protected_operations.start(
                contract,
                invocation,
                provider=_Provider(),
            ) as operation:
                result = operation.call(
                    ProviderPhase("read", information_flow=True),
                    lambda selected=selected: selected,
                )
                assert operation.complete(result, _evidence(pid)) is selected

        assert len(observations) == 7
        assert observations[0].result_descriptor["result_type"] == "opaque"
        assert _HOST_RESULT_SENTINEL not in json.dumps(
            observations[0].result_descriptor,
            sort_keys=True,
        )
        assert [
            item.result_descriptor["digest_mode"] for item in observations
        ] == [
            "digest_unavailable",
            "digest_unavailable",
            "digest_unavailable",
            "digest_unavailable",
            "digest_unavailable",
            "digest_unavailable",
            "canonical_bounded",
        ]
        assert all(item.result_sha256 is None for item in observations[:-1])
        assert observations[-1].result_sha256 == hashlib.sha256(
            json.dumps(
                {
                    "bytes_sha256": hashlib.sha256(binary).hexdigest(),
                    "size_bytes": len(binary),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        assert all(item.data_labels == labels for item in observations)
        assert all(
            item.data_flow_direction == DataFlowDirection.INGRESS.value
            for item in observations
        )
        assert all(item.source_refs_sha256 == DataFlowContext(
            labels=labels,
            source_refs=(source,),
        ).source_refs_hash() for item in observations)
        assert all(
            len(json.dumps(item.result_descriptor)) < 4096
            for item in observations
        )
        assert metaclass_hook_calls == 0
        assert metaclass_comparison_calls == 0
        assert property_hook_calls == 0
        retained = json.dumps(
            {
                "effects": to_jsonable(runtime.store.list_external_effects(pid=pid)),
                "events": to_jsonable(runtime.store.list_events()),
                "audit": to_jsonable(runtime.store.list_audit()),
                "semantic_api": runtime.semantic.status(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        assert _HOST_RESULT_SENTINEL not in retained


def test_opaque_result_type_name_never_reaches_shadow_ledger_or_api() -> None:
    sentinel = "SEMANTIC_RESULT_TYPE_SECRET_SENTINEL"
    opaque_type = type(sentinel, (), {})
    config = AgentLibOSConfig(
        semantic=SemanticDefaults(mode="shadow", adapter="deterministic")
    )
    with TemporaryDirectory() as temp_dir:
        runtime = Runtime.open(
            "local",
            config=config,
            substrate=LocalResourceProviderSubstrate(Path(temp_dir)),
        )
        try:
            pid = runtime.process.spawn(goal="opaque provider result fixture")
            runtime.capability.issue_trusted(
                pid,
                "test:item",
                [CapabilityRight.READ],
                issued_by="test",
            )
            decision = runtime.capability.require(
                pid,
                "test:item",
                CapabilityRight.READ,
                consume=False,
            )
            contract = ProtectedOperationContract(
                name="primitive.filesystem.opaque_result_fixture",
                provider="test",
                operation="read",
                evidence_roles=("audit", "event", "effect"),
                resource_policy=ResourcePolicy.NONE,
                information_flow=True,
            )
            runtime.protected_operations.register_contract(contract)
            observed: list[object] = []
            invocation = ProtectedOperationInvocation(
                pid=pid,
                actor=pid,
                target="test:item",
                decisions=(decision,),
                canonical_args={"item_sha256": _A},
                observation={"item_sha256": _A},
                post_commit_result_observer=(
                    lambda _result, observation: observed.append(observation)
                ),
            )
            opaque = opaque_type()
            with runtime.protected_operations.start(
                contract,
                invocation,
                provider=_Provider(),
            ) as operation:
                result = operation.call(
                    ProviderPhase("read", information_flow=True),
                    lambda: opaque,
                )
                assert operation.complete(result, _evidence(pid)) is opaque

            assert len(observed) == 1
            assert observed[0].result_descriptor["result_type"] == "opaque"
            surfaces = {
                "descriptor": observed[0].result_descriptor,
                "semantic_status": runtime.semantic.status(),
                "semantic_api": runtime.semantic.query_assessments(limit=100),
                "semantic_jobs": [
                    dict(row)
                    for row in runtime.store._query(  # noqa: SLF001 - privacy oracle
                        "SELECT projection_json, bindings_json, error_code "
                        "FROM semantic_assessment_jobs"
                    )
                ],
                "semantic_assessments": [
                    dict(row)
                    for row in runtime.store._query(  # noqa: SLF001 - privacy oracle
                        "SELECT record_json FROM semantic_assessments"
                    )
                ],
                "effects": runtime.store.list_external_effects(pid=pid),
                "events": runtime.store.list_events(),
                "audit": runtime.store.list_audit(),
            }
            assert sentinel not in json.dumps(
                to_jsonable(surfaces),
                ensure_ascii=False,
                default=str,
                sort_keys=True,
            )
        finally:
            runtime.close()


def test_frozen_provider_registry_and_tool_schema_reach_shadow_assessment() -> None:
    config = AgentLibOSConfig(
        semantic=SemanticDefaults(mode="shadow", adapter="deterministic")
    )
    with TemporaryDirectory() as temp_dir:
        runtime = Runtime.open(
            "local",
            config=config,
            substrate=LocalResourceProviderSubstrate(Path(temp_dir)),
        )
        try:
            pid = runtime.process.spawn(goal="provider registry provenance fixture")
            runtime.capability.issue_trusted(
                pid,
                "test:item",
                [CapabilityRight.READ],
                issued_by="test",
            )
            decision = runtime.capability.require(
                pid,
                "test:item",
                CapabilityRight.READ,
                consume=False,
            )
            contract = ProtectedOperationContract(
                name="primitive.jsonrpc.registry_provenance_fixture",
                provider="jsonrpc",
                operation="call",
                evidence_roles=("audit", "event", "effect"),
                resource_policy=ResourcePolicy.NONE,
                information_flow=True,
            )
            runtime.protected_operations.register_contract(contract)
            registry = ProviderRegistryBinding(
                registry_spec_sha256=_A,
                registry_generation=7,
            )
            observed: list[object] = []
            invocation = ProtectedOperationInvocation(
                pid=pid,
                actor=pid,
                target="test:item",
                decisions=(decision,),
                canonical_args={"item_sha256": _A},
                observation={"item_sha256": _A, "tool_schema_sha256": _B},
                provider_registry_binding=registry,
                provider_registry_binding_resolver=lambda: registry,
                provider_registry_phase_guard=lambda: threading.RLock(),
                post_commit_result_observer=(
                    lambda _result, observation: observed.append(observation)
                ),
            )
            with runtime.protected_operations.start(
                contract,
                invocation,
                provider=_Provider(),
            ) as operation:
                result = operation.call(
                    ProviderPhase("call", information_flow=True),
                    lambda: {"ok": True},
                )
                assert operation.complete(result, _evidence(pid)) == {"ok": True}

            assert len(observed) == 1
            effect_id = observed[0].effect_id
            deadline = time.monotonic() + 5
            records = ()
            while time.monotonic() < deadline:
                records = runtime.uow.semantic.query_semantic_assessments(
                    after=None,
                    limit=10,
                    pid=pid,
                ).records
                records = tuple(
                    record for record in records if record.effect_id == effect_id
                )
                if records:
                    break
                time.sleep(0.01)
            assert len(records) == 1
            assert records[0].provider_spec_sha256 == _A
            assert records[0].tool_schema_sha256 == _B
        finally:
            runtime.close()


def test_real_git_status_result_is_captured_as_provider_ingress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    git_home = tmp_path / "git-home"
    repository.mkdir()
    git_home.mkdir()
    monkeypatch.setenv("HOME", str(git_home))

    def git(*args: str) -> None:
        environment = dict(os.environ)
        environment.update(
            {
                "GIT_TRACE": "0",
                "GIT_TRACE2": "0",
                "GIT_TRACE2_EVENT": "0",
                "GIT_TRACE2_PERF": "0",
            }
        )
        subprocess.run(
            ["git", *args],
            cwd=repository,
            env=environment,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    git("init", "-q")
    git("symbolic-ref", "HEAD", "refs/heads/main")
    git("config", "user.name", "Agent libOS Semantic Test")
    git("config", "user.email", "semantic@example.test")
    (repository / "tracked.txt").write_text("initial\n", encoding="utf-8")
    git("add", "--", "tracked.txt")
    git("commit", "-q", "-m", "initial")
    (repository / "untracked.txt").write_text("provider ingress\n", encoding="utf-8")

    runtime = Runtime.open(
        ":memory:",
        config=AgentLibOSConfig(
            semantic=SemanticDefaults(mode="shadow", adapter="deterministic")
        ),
        substrate=LocalResourceProviderSubstrate(repository),
    )
    try:
        pid = runtime.process.spawn(goal="inspect a real Git provider result")
        runtime.capability.issue_trusted(
            pid,
            "git:workspace",
            [CapabilityRight.READ, CapabilityRight.DIFF],
            issued_by="test",
        )
        runtime.filesystem.grant_directory(
            pid,
            ".",
            [CapabilityRight.READ],
            issued_by="test",
        )
        status = runtime.git.status(pid)
        expected_digest, descriptor = _post_commit_result_identity(status)
        assert expected_digest is not None
        assert descriptor["digest_mode"] == "canonical_bounded"

        deadline = time.monotonic() + 5
        matching = ()
        while time.monotonic() < deadline:
            records = runtime.uow.semantic.query_semantic_assessments(
                after=None,
                limit=100,
                pid=pid,
                domain="git",
            ).records
            matching = tuple(
                record
                for record in records
                if record.kind == "provider_ingress"
                and record.input_sha256 == expected_digest
            )
            if matching:
                break
            time.sleep(0.01)
        assert len(matching) == 1
        assert matching[0].effect_id is not None
        assert matching[0].provider_spec_sha256 is not None
    finally:
        runtime.close()


def _git_result_state() -> GitStateToken:
    return GitStateToken(
        token=_A,
        repository_identity="repository:test",
        worktree_id="main",
        head_ref=None,
        head_oid=None,
        index_sha256=_A,
        config_sha256=_B,
        refs_sha256=_C,
        worktrees_sha256=_A,
        pull_requests_sha256=_B,
        worktree_sha256=_C,
    )


def _host_result_digest_cases(marker: str) -> dict[str, object]:
    shell_stdout = f"{marker}:" + ("密" * 100_000)
    patch = f"{marker}\n" + ("x" * 300_000)
    patch_bytes = patch.encode("utf-8")
    patch_path_bytes = f"{marker}/patch.txt".encode("utf-8")
    patch_path = GitPath(
        display=patch_path_bytes.decode("utf-8"),
        path_b64=base64.b64encode(patch_path_bytes).decode("ascii"),
    )
    status_entries: list[GitStatusEntry] = []
    status_manifest = hashlib.sha256()
    for index in range(1_100):
        raw_path = f"{marker}/status-{index}.txt".encode("utf-8")
        status_manifest.update(len(raw_path).to_bytes(8, "big"))
        status_manifest.update(raw_path)
        status_entries.append(
            GitStatusEntry(
                path=GitPath(
                    display=raw_path.decode("utf-8"),
                    path_b64=base64.b64encode(raw_path).decode("ascii"),
                ),
                kind=GitStatusKind.UNTRACKED,
                index_status="?",
                worktree_status="?",
            )
        )
    return {
        "filesystem_path_state": PathState(
            exists=True,
            kind="file",
            size_bytes=len(marker.encode("utf-8")),
            modified_at=marker,
        ),
        "shell_large_stdout": CommandResult(
            argv=["semantic-result-fixture", marker],
            returncode=0,
            stdout=shell_stdout,
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
            metrics=CommandMetrics(
                wall_seconds=0.01,
                cpu_seconds=0.005,
                peak_memory_bytes=4_096,
            ),
        ),
        "git_slots_large_patch": GitDiffResult(
            repository_id="repository:test",
            worktree_id="main",
            scope="worktree",
            base_oid=None,
            head_oid=None,
            patch=patch,
            patch_b64=base64.b64encode(patch_bytes).decode("ascii"),
            changed_paths=[patch_path],
            state=_git_result_state(),
            truncated=False,
            bytes=len(patch_bytes),
            sha256=hashlib.sha256(patch_bytes).hexdigest(),
        ),
        "git_slots_large_status": GitStatusResult(
            repository_id="repository:test",
            worktree_id="main",
            branch=None,
            upstream=None,
            ahead=0,
            behind=0,
            head_oid=None,
            entries=status_entries,
            state=_git_result_state(),
            truncated=False,
            bytes=sum(len(item.path.display.encode("utf-8")) for item in status_entries),
            sha256=status_manifest.hexdigest(),
        ),
        "jsonrpc_nested_body": JsonRpcCallResult(
            endpoint_id="semantic-fixture",
            method_id="read",
            rpc_method="fixture.read",
            request_id=f"request-{marker}",
            status=JsonRpcCallStatus.OK,
            http_status=200,
            ok=True,
            result={"payload": {"text": marker, "values": [1, 2, 3]}},
            response_bytes=len(marker.encode("utf-8")),
            duration_s=0.01,
        ),
        "mcp_nested_body": McpCallResult(
            server_id="semantic-fixture",
            tool_id="read",
            mcp_name="fixture.read",
            status=McpCallStatus.OK,
            ok=True,
            result={
                "content": [{"type": "text", "text": marker}],
                "structured_content": {"text": marker},
            },
            response_bytes=len(marker.encode("utf-8")),
            duration_s=0.01,
        ),
    }


def test_host_provider_result_digest_matrix_is_bounded_complete_and_payload_free() -> None:
    first_cases = _host_result_digest_cases(f"{_HOST_RESULT_SENTINEL}:first")
    second_cases = _host_result_digest_cases(f"{_HOST_RESULT_SENTINEL}:second")

    assert first_cases.keys() == second_cases.keys()
    for case_name, first in first_cases.items():
        second = second_cases[case_name]
        first_digest, first_descriptor = _post_commit_result_identity(first)
        second_digest, second_descriptor = _post_commit_result_identity(second)

        assert first_digest is not None, case_name
        assert second_digest is not None, case_name
        assert first_digest != second_digest, case_name
        assert first_descriptor["digest_mode"] != "digest_unavailable", case_name
        assert second_descriptor["digest_mode"] != "digest_unavailable", case_name
        assert first_descriptor["result_type"] == (
            f"{type(first).__module__}.{type(first).__qualname__}"
        )
        assert second_descriptor["result_type"] == (
            f"{type(second).__module__}.{type(second).__qualname__}"
        )
        first_descriptor_json = json.dumps(
            first_descriptor,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
        second_descriptor_json = json.dumps(
            second_descriptor,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
        assert _HOST_RESULT_SENTINEL not in first_descriptor_json, case_name
        assert _HOST_RESULT_SENTINEL not in second_descriptor_json, case_name
        assert len(first_descriptor_json.encode("utf-8")) < 4_096, case_name
        assert len(second_descriptor_json.encode("utf-8")) < 4_096, case_name

    class OpaqueProviderResult:
        def to_dict(self) -> dict[str, str]:
            raise AssertionError("opaque provider hooks must not run")

    class HostileMapping(dict[str, object]):
        def items(self):
            raise AssertionError("provider mapping hooks must not run")

    for opaque in (OpaqueProviderResult(), HostileMapping(payload="hidden")):
        digest, descriptor = _post_commit_result_identity(opaque)
        assert digest is None
        assert descriptor["digest_mode"] == "digest_unavailable"
        assert _HOST_RESULT_SENTINEL not in json.dumps(descriptor)


def test_git_internal_flow_snapshot_digest_ignores_callable_fields() -> None:
    resolver_calls = 0

    def forbidden_resolver():
        nonlocal resolver_calls
        resolver_calls += 1
        raise AssertionError("Git result resolver must not be invoked")

    flow = _GitFlowSnapshot(
        context=DataFlowContext(),
        state_version="flow-state-v1",
        state_version_resolver=forbidden_resolver,
    )
    read = _GitReadFlowSnapshot(
        flow=flow,
        repository_state_token="repository-state-v1",
        refresh=forbidden_resolver,
    )
    observed_text: list[str | bytes] = []
    flow_digest, _flow_descriptor = _post_commit_result_identity(flow)
    read_digest, _read_descriptor = _post_commit_result_identity(read)
    visit_bounded_host_result_text(
        read,
        contract_name="primitive.git.read",
        visitor=observed_text.append,
    )

    assert flow_digest is not None
    assert read_digest is not None and read_digest != flow_digest
    assert "flow-state-v1" in observed_text
    assert "repository-state-v1" in observed_text
    assert resolver_calls == 0


def _semantic_call(
    pid: str,
    sink: DataSink,
    *,
    payload: object,
    labels: DataLabels | None = None,
    flow_context: DataFlowContext | None = None,
) -> ProtectedSemanticAssessmentCall:
    return ProtectedSemanticAssessmentCall(
        pid=pid,
        actor=pid,
        profile_id="classifier",
        profile_identity_sha256=_A,
        classifier_id="semantic-test",
        classifier_version="1",
        classifier_artifact_sha256=_B,
        projection_sha256=_C,
        response_schema_sha256=_B,
        deadline_at="2027-01-01T00:00:00+00:00",
        sink=sink,
        data_flow_context=(
            flow_context
            if flow_context is not None
            else DataFlowContext(labels=labels or DataLabels())
        ),
        egress_payload=payload,
    )


def test_semantic_protected_port_records_only_digests_and_suppresses_recursion() -> None:
    sentinel = "SEMANTIC_PROTECTED_PAYLOAD_SENTINEL"
    source_identity_sentinel = "SEMANTIC_SOURCE_IDENTITY_SECRET_SENTINEL_8f91"
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(goal="protected semantic classifier")
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {"value": "source evidence"},
            metadata=ObjectMetadata(origin=source_identity_sentinel),
        )
        flow_context = runtime.data_flow.context_from_source_oids(
            pid,
            [source.oid],
        )
        sink = DataSink("llm:classifier", identity_sha256=_A)
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=sink.identity,
                identity_sha256=_A,
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="secret",
            ),
            actor="test.host",
            require_capability=False,
        )
        globally_observed: list[tuple[object, object]] = []
        runtime.protected_operations.bind_post_commit_result_observer(
            lambda result, observation: globally_observed.append(
                (result, observation)
            )
        )
        port = SdkProtectedSemanticCallPort(
            runtime.protected_operations,
            profile_identity_resolver=lambda profile_id: _A
            if profile_id == "classifier"
            else _B,
        )
        dispatch_count = 0

        def dispatch() -> dict[str, str]:
            nonlocal dispatch_count
            dispatch_count += 1
            return {"provider_value": sentinel}

        result = port.invoke(
            _semantic_call(
                pid,
                sink,
                payload={"redacted_intent": sentinel},
                flow_context=flow_context,
            ),
            provider=_Provider(),
            dispatch=dispatch,
        )

        assert result == {"provider_value": sentinel}
        assert dispatch_count == 1
        assert len(globally_observed) == 1
        assert globally_observed[0][0] == result
        observed_labels = globally_observed[0][1].data_labels
        assert observed_labels.sensitivity.value == "normal"
        assert observed_labels.trust_level.value == "untrusted"
        assert observed_labels.integrity.value == "untrusted"
        assert observed_labels.origin == "external:llm"
        effects = runtime.store.list_external_effects(pid=pid)
        assert effects[-1].provider == "llm"
        assert effects[-1].operation == "complete"
        data_flow_evidence = effects[-1].provider_metadata["data_flow"]
        assert data_flow_evidence["source_refs_sha256"] == (
            flow_context.source_refs_hash()
        )
        assert data_flow_evidence["source_ref_count"] == 1
        assert "source_refs" not in data_flow_evidence
        assert data_flow_evidence["labels"] == {
            "sensitivity": "normal",
            "integrity": "unknown",
            "trust_level": "unknown",
            "identity_present": False,
            "identity_mixed": False,
        }
        decisions = [
            item
            for item in runtime.store.list_data_flow_decisions(pid=pid)
            if item.sink == sink.identity
        ]
        assert decisions
        assert all(item.source_refs == () for item in decisions)
        assert all(
            item.labels.origin is None
            and item.labels.tenant is None
            and item.labels.principal is None
            and item.labels.declassification_authority is None
            for item in decisions
        )
        decision_ids = {item.decision_id for item in decisions}
        decision_events = [
            item
            for item in runtime.store.list_events()
            if item.type == EventType.DATA_FLOW_DECISION
            and item.payload.get("decision_id") in decision_ids
        ]
        decision_audits = [
            item
            for item in runtime.store.list_audit()
            if item.action == "data_flow.egress"
            and (item.decision or {}).get("decision_id") in decision_ids
        ]
        assert decision_events and decision_audits
        assert all("source_refs" not in item.payload for item in decision_events)
        assert all(item.input_refs == [] for item in decision_audits)
        persisted_decisions = runtime.store._query(  # noqa: SLF001 - privacy oracle
            "SELECT labels_json, source_refs_json FROM data_flow_decisions "
            "WHERE pid = ? AND sink = ?",
            (pid, sink.identity),
        )
        assert persisted_decisions
        assert all(
            json.loads(item["source_refs_json"]) == []
            and json.loads(item["labels_json"])["origin"] is None
            for item in persisted_decisions
        )
        retained = json.dumps(
            {
                "effect": effects[-1].provider_metadata,
                "audit": [item.__dict__ for item in decision_audits],
                "events": [item.__dict__ for item in decision_events],
                "decision_rows": [dict(item) for item in persisted_decisions],
            },
            default=str,
            sort_keys=True,
        )
        assert sentinel not in retained
        assert source.oid not in retained
        assert source_identity_sentinel not in retained


def test_semantic_protected_port_never_creates_data_release_request() -> None:
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(goal="blocked semantic classifier")
        sink = DataSink("llm:classifier", identity_sha256=_A)
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=sink.identity,
                identity_sha256=_A,
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity="secret",
            ),
            actor="test.host",
            require_capability=False,
        )
        port = SdkProtectedSemanticCallPort(
            runtime.protected_operations,
            profile_identity_resolver=lambda _profile_id: _A,
        )
        dispatched: list[bool] = []

        with pytest.raises(CapabilityDenied, match="data release"):
            port.invoke(
                _semantic_call(
                    pid,
                    sink,
                    payload={"metadata": "only"},
                    labels=DataLabels(sensitivity="secret"),
                ),
                provider=_Provider(),
                dispatch=lambda: dispatched.append(True),
            )

        assert dispatched == []
        assert runtime.human.pending() == []
        assert runtime.store.list_external_effects(pid=pid) == []


def test_semantic_contract_rejects_caller_override_of_source_ref_redaction() -> None:
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(goal="semantic evidence redaction is Host-owned")
        SdkProtectedSemanticCallPort(
            runtime.protected_operations,
            profile_identity_resolver=lambda _profile_id: _A,
        )
        invocation = ProtectedOperationInvocation(
            pid=pid,
            actor=pid,
            target="llm:classifier",
            data_flow_redact_source_refs_evidence=False,
        )

        with pytest.raises(ValidationError, match="redaction is Host-mandated"):
            runtime.protected_operations.start(
                "semantic.llm.assess",
                invocation,
                provider=_Provider(),
            )


def test_semantic_classifier_stale_source_ref_blocks_before_provider_and_stays_redacted() -> None:
    source_identity_sentinel = "SEMANTIC_STALE_SOURCE_SECRET_SENTINEL_45ab"
    with temporary_runtime() as runtime:
        pid = runtime.process.spawn(goal="block stale semantic classifier sources")
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {"value": "first version"},
            metadata=ObjectMetadata(origin=source_identity_sentinel),
            immutable=False,
        )
        stale_flow = runtime.data_flow.context_from_source_oids(pid, [source.oid])
        runtime.memory.update_object(
            pid,
            source,
            ObjectPatch(payload={"value": "second version"}),
            expected_version=1,
        )
        sink = DataSink("llm:classifier", identity_sha256=_A)
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=sink.identity,
                identity_sha256=_A,
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="secret",
            ),
            actor="test.host",
            require_capability=False,
        )
        port = SdkProtectedSemanticCallPort(
            runtime.protected_operations,
            profile_identity_resolver=lambda _profile_id: _A,
        )
        provider_calls: list[bool] = []

        with pytest.raises(CapabilityDenied):
            port.invoke(
                _semantic_call(
                    pid,
                    sink,
                    payload={"projection_mode": "metadata_only"},
                    flow_context=stale_flow,
                ),
                provider=_Provider(),
                dispatch=lambda: provider_calls.append(True),
            )

        assert provider_calls == []
        decisions = [
            item
            for item in runtime.store.list_data_flow_decisions(
                pid=pid,
                outcome="deny",
            )
            if item.sink == sink.identity
        ]
        assert len(decisions) == 1
        decision = decisions[0]
        assert decision.source_refs == ()
        assert source.oid not in decision.reason
        events = [
            item
            for item in runtime.store.list_events()
            if item.type == EventType.DATA_FLOW_DECISION
            and item.payload.get("decision_id") == decision.decision_id
        ]
        audits = [
            item
            for item in runtime.store.list_audit()
            if item.action == "data_flow.egress"
            and (item.decision or {}).get("decision_id") == decision.decision_id
        ]
        retained = json.dumps(
            {
                "decision": decision.__dict__,
                "events": [item.__dict__ for item in events],
                "audits": [item.__dict__ for item in audits],
            },
            default=str,
            sort_keys=True,
        )
        assert events and audits
        assert events[0].payload["source_refs_sha256"] == (
            stale_flow.source_refs_hash()
        )
        assert events[0].payload["source_ref_count"] == 1
        assert "source_refs" not in events[0].payload
        assert audits[0].input_refs == []
        assert source.oid not in retained
        assert source_identity_sentinel not in retained


def _approval_request(
    pid: str,
    request_id: str,
    *,
    tenant: str | None = None,
) -> HumanRequest:
    resource = "filesystem:workspace:reports/item.txt"
    context = {
        "adapter": "filesystem",
        "operation": "read_text",
        "authority_operation": "filesystem.read",
        "pid": pid,
        "resource": resource,
        "right": "read",
    }
    binding = {
        "effect_id": f"eff_{request_id}",
        "canonical_args_hash": canonical_effect_hash(context),
        "target_state_version": None,
    }
    labels = DataLabels(tenant=tenant, principal="principal-a" if tenant else None)
    now = utc_now()
    return HumanRequest(
        request_id=request_id,
        pid=pid,
        human="host",
        payload={
            "type": "external_operation_approval",
            "_agent_libos_authority_request_origin": "external_operation",
            "context": context,
            "effect_binding": binding,
            "requested_once_capability": {
                "subject": pid,
                "resource": resource,
                "rights": ["read"],
                "constraints": {"approval_binding": dict(binding)},
            },
            "_agent_libos_data_flow_context": DataFlowContext(
                labels=labels
            ).to_dict(),
        },
        status=HumanRequestStatus.PENDING,
        decision=None,
        blocking=True,
        created_at=now,
        updated_at=now,
    )


def _semantic_pid(runtime: Runtime) -> str:
    return runtime.process.spawn(
        goal="semantic exact read fixture",
        authority_manifest={
            "approval_policy": {
                "semantic_auto_approval": {
                    "schema_version": 1,
                    "rules": [
                        {
                            "rule_id": "exact-read",
                            "authority_operation": "filesystem.read",
                            "resource": "filesystem:workspace:reports/*",
                            "rights": ["read"],
                        }
                    ],
                }
            }
        },
    )


def _assessment_for(runtime: Runtime, request_id: str) -> object:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        records = runtime.uow.semantic.query_semantic_assessments(
            after=None,
            limit=100,
            request_id=request_id,
        ).records
        if records:
            return records[0]
        time.sleep(0.01)
    raise AssertionError(f"semantic assessment did not complete: {request_id}")


def _drain_semantic_synchronously(runtime: Runtime) -> None:
    assert runtime.semantic.shutdown()
    while runtime.semantic.process_one():
        pass


def test_malformed_persisted_approval_matrix_is_durable_private_and_observational(
    monkeypatch,
) -> None:
    sentinel = "SEMANTIC_MALFORMED_APPROVAL_SECRET_SENTINEL_7c294f"
    config = AgentLibOSConfig(
        semantic=SemanticDefaults(mode="shadow", adapter="deterministic")
    )
    with TemporaryDirectory() as temp_dir:
        runtime = Runtime.open(
            "local",
            config=config,
            substrate=LocalResourceProviderSubstrate(Path(temp_dir)),
        )
        try:
            pid = _semantic_pid(runtime)
            _drain_semantic_synchronously(runtime)
            provider_calls: list[object] = []

            class _FailIfCalled:
                def assess(self, request: object) -> object:
                    provider_calls.append(request)
                    raise AssertionError(
                        "malformed approval must terminate before provider dispatch"
                    )

            runtime.semantic._assessor = _FailIfCalled()
            mutations = (
                ("context-missing", lambda payload: payload.pop("context")),
                ("context-type", lambda payload: payload.__setitem__("context", [])),
                (
                    "capability-missing",
                    lambda payload: payload.pop("requested_once_capability"),
                ),
                (
                    "capability-type",
                    lambda payload: payload.__setitem__(
                        "requested_once_capability", []
                    ),
                ),
                ("binding-missing", lambda payload: payload.pop("effect_binding")),
                (
                    "binding-type",
                    lambda payload: payload.__setitem__("effect_binding", []),
                ),
                (
                    "action",
                    lambda payload: payload["context"].__setitem__(
                        "authority_operation", "Filesystem.READ"
                    ),
                ),
                (
                    "resource",
                    lambda payload: payload["requested_once_capability"].__setitem__(
                        "resource", 7
                    ),
                ),
                (
                    "rights-type",
                    lambda payload: payload["requested_once_capability"].__setitem__(
                        "rights", "read"
                    ),
                ),
                (
                    "rights-item",
                    lambda payload: payload["requested_once_capability"].__setitem__(
                        "rights", [7]
                    ),
                ),
                (
                    "data-flow",
                    lambda payload: payload.__setitem__(
                        "_agent_libos_data_flow_context", {"labels": "invalid"}
                    ),
                ),
                (
                    "optional-digest",
                    lambda payload: payload["context"].__setitem__(
                        "sink_identity_sha256", sentinel
                    ),
                ),
            )
            jobs = []
            request_ids: list[str] = []
            for name, mutate in mutations:
                request = deepcopy(_approval_request(pid, f"malformed-{name}"))
                request.payload["private_probe"] = sentinel
                mutate(request.payload)
                runtime.human.requests.insert(request)
                captured = runtime.semantic.capture_approval(request)
                assert captured is not None, name
                jobs.append(captured)
                request_ids.append(request.request_id)

            candidate_request = deepcopy(
                _approval_request(pid, "malformed-authority-candidate")
            )
            candidate_request.payload["private_probe"] = sentinel
            runtime.human.requests.insert(candidate_request)
            candidate_reader = (
                runtime.semantic._authority.semantic_auto_approval_candidate
            )
            monkeypatch.setattr(
                runtime.semantic._authority,
                "semantic_auto_approval_candidate",
                lambda *_args, **_kwargs: {
                    "schema_version": 1,
                    "rule_id": "missing-required-candidate-fields",
                },
            )
            captured_candidate = runtime.semantic.capture_approval(candidate_request)
            monkeypatch.setattr(
                runtime.semantic._authority,
                "semantic_auto_approval_candidate",
                candidate_reader,
            )
            assert captured_candidate is not None
            jobs.append(captured_candidate)
            request_ids.append(candidate_request.request_id)

            for job, request_id in zip(jobs, request_ids, strict=True):
                assert job.kind == "approval"
                assert job.domain == "runtime"
                assert job.pid == pid
                assert job.request_id == request_id
                assert job.operation_id is None
                assert job.effect_id is None
                assert job.projection["action_id"] == (
                    "runtime.malformed_external_operation"
                )
                assert job.projection["candidate"] is None
                assert job.projection["hard_violations"] == []
                assert not any(job.projection["features"].values())
                assert job.bindings["input_sha256"] == (
                    job.projection["input_sha256"]
                )
                assert sentinel not in json.dumps(
                    to_jsonable(job), ensure_ascii=False, default=str
                )

            human_before = tuple(runtime.human.list(pid))
            capabilities_before = to_jsonable(runtime.capability.list_subject(pid))
            manifest_before = to_jsonable(
                runtime.authority_manifests.get_for_process(pid)
            )
            effects_before = to_jsonable(runtime.store.list_external_effects(pid=pid))
            process_before = to_jsonable(runtime.uow.processes.get_process(pid))

            for _job in jobs:
                assert runtime.semantic.process_one()

            assert provider_calls == []
            assert tuple(runtime.human.list(pid)) == human_before
            assert all(
                request.status is HumanRequestStatus.PENDING
                for request in runtime.human.pending()
                if request.request_id in request_ids
            )
            assert to_jsonable(runtime.capability.list_subject(pid)) == (
                capabilities_before
            )
            assert to_jsonable(runtime.authority_manifests.get_for_process(pid)) == (
                manifest_before
            )
            assert to_jsonable(runtime.store.list_external_effects(pid=pid)) == (
                effects_before
            )
            assert to_jsonable(runtime.uow.processes.get_process(pid)) == process_before

            for queued, request_id in zip(jobs, request_ids, strict=True):
                terminal = runtime.uow.semantic.get_semantic_assessment_job(
                    queued.job_id
                )
                assert terminal is not None
                assert terminal.status.value == "failed"
                assert terminal.error_code == "invalid_schema"
                assert terminal.projection == {}
                assert terminal.projection_retention.value == "hash_only"
                record = _assessment_for(runtime, request_id)
                assert record.kind == "approval"
                assert record.domain == "runtime"
                assert record.action_id == "runtime.malformed_external_operation"
                assert record.status == "invalid_schema"
                assert record.shadow_outcome == "require_human"
                assert record.reason_codes == ("schema_invalid",)
                assert record.proven_predicates == ()

            api_page = runtime.semantic.query_assessments(pid=pid, limit=100)
            api_details = [
                runtime.semantic.get_assessment(item["assessment_id"])
                for item in api_page["items"]
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
            retained_surfaces = {
                "semantic_status": runtime.semantic.status(),
                "semantic_api_page": api_page,
                "semantic_api_details": api_details,
                "semantic_sql": raw_semantic_rows,
                "events": runtime.events.list(),
                "audit": runtime.audit.trace(),
            }
            assert sentinel not in json.dumps(
                to_jsonable(retained_surfaces),
                ensure_ascii=False,
                default=str,
                sort_keys=True,
            )
        finally:
            runtime.close()


def test_malformed_capture_infrastructure_and_uncanonical_fail_without_assessment(
    monkeypatch,
) -> None:
    config = AgentLibOSConfig(
        semantic=SemanticDefaults(mode="shadow", adapter="deterministic")
    )
    with TemporaryDirectory() as temp_dir:
        runtime = Runtime.open(
            "local",
            config=config,
            substrate=LocalResourceProviderSubstrate(Path(temp_dir)),
        )
        try:
            pid = _semantic_pid(runtime)
            _drain_semantic_synchronously(runtime)
            failure_count = runtime.semantic.status()["queue"]["capture_failures"]

            authority_failure = _approval_request(pid, "authority-reader-failure")
            runtime.human.requests.insert(authority_failure)
            authority_reader = runtime.semantic._authority.get_for_process
            monkeypatch.setattr(
                runtime.semantic._authority,
                "get_for_process",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("authority repository unavailable")
                ),
            )
            assert runtime.semantic.capture_approval(authority_failure) is None
            monkeypatch.setattr(
                runtime.semantic._authority,
                "get_for_process",
                authority_reader,
            )
            failure_count += 1
            assert runtime.semantic.status()["queue"]["capture_failures"] == (
                failure_count
            )

            enqueue_failure = _approval_request(pid, "enqueue-runtime-failure")
            runtime.human.requests.insert(enqueue_failure)
            enqueue = runtime.semantic._repository.enqueue_semantic_assessment_job
            monkeypatch.setattr(
                runtime.semantic._repository,
                "enqueue_semantic_assessment_job",
                lambda _record: (_ for _ in ()).throw(
                    RuntimeError("semantic repository unavailable")
                ),
            )
            assert runtime.semantic.capture_approval(enqueue_failure) is None
            failure_count += 1
            assert runtime.semantic.status()["queue"]["capture_failures"] == (
                failure_count
            )

            enqueue_calls = 0

            def reject_enqueue(_record: object) -> object:
                nonlocal enqueue_calls
                enqueue_calls += 1
                raise ValidationError("semantic repository rejected job")

            enqueue_validation = _approval_request(
                pid, "enqueue-validation-failure"
            )
            enqueue_validation.payload.pop("context")
            runtime.human.requests.insert(enqueue_validation)
            monkeypatch.setattr(
                runtime.semantic._repository,
                "enqueue_semantic_assessment_job",
                reject_enqueue,
            )
            assert runtime.semantic.capture_approval(enqueue_validation) is None
            assert enqueue_calls == 1
            failure_count += 1
            assert runtime.semantic.status()["queue"]["capture_failures"] == (
                failure_count
            )
            monkeypatch.setattr(
                runtime.semantic._repository,
                "enqueue_semantic_assessment_job",
                enqueue,
            )

            uncanonical = _approval_request(pid, "uncanonical-approval")
            uncanonical.payload["not_json"] = float("nan")
            assert runtime.semantic.capture_approval(uncanonical) is None
            failure_count += 1
            assert runtime.semantic.status()["queue"]["capture_failures"] == (
                failure_count
            )

            for request_id in (
                authority_failure.request_id,
                enqueue_failure.request_id,
                enqueue_validation.request_id,
                uncanonical.request_id,
            ):
                assert runtime.uow.semantic.query_semantic_assessments(
                    after=None,
                    limit=100,
                    request_id=request_id,
                ).records == ()
                assert not any(
                    job.request_id == request_id
                    for job in runtime.uow.semantic.query_semantic_assessment_jobs(
                        statuses=("queued", "claimed", "failed", "succeeded"),
                        projection_expires_before=None,
                        limit=100,
                    )
                )
        finally:
            runtime.close()


def test_approval_binding_tamper_is_fail_closed_without_authority_delta() -> None:
    config = AgentLibOSConfig(
        semantic=SemanticDefaults(mode="shadow", adapter="deterministic")
    )
    with TemporaryDirectory() as temp_dir:
        runtime = Runtime.open(
            "local",
            config=config,
            substrate=LocalResourceProviderSubstrate(Path(temp_dir)),
        )
        try:
            pid = _semantic_pid(runtime)
            runtime.semantic._human_outcome_reader = lambda _request_id: "pending"
            before = tuple(runtime.capability.list_subject(pid))

            def capture(request: HumanRequest) -> None:
                runtime.human.requests.insert(request)
                assert runtime.semantic.capture_approval(request) is not None

            valid = _approval_request(pid, "binding-valid")
            capture(valid)

            wrong_hash = deepcopy(valid)
            wrong_hash.request_id = "binding-hash-tamper"
            wrong = "f" * 64
            wrong_hash.payload["effect_binding"]["canonical_args_hash"] = wrong
            wrong_hash.payload["requested_once_capability"]["constraints"][
                "approval_binding"
            ]["canonical_args_hash"] = wrong
            capture(wrong_hash)

            nested = deepcopy(valid)
            nested.request_id = "binding-nested-tamper"
            nested.payload["requested_once_capability"]["constraints"][
                "approval_binding"
            ]["effect_id"] = "eff_other"
            capture(nested)

            subject = deepcopy(valid)
            subject.request_id = "binding-subject-tamper"
            subject.payload["requested_once_capability"]["subject"] = "pid_other"
            capture(subject)

            context_pid = deepcopy(valid)
            context_pid.request_id = "binding-context-pid-tamper"
            context_pid.payload["context"]["pid"] = "pid_other"
            changed_binding = {
                "effect_id": "eff_context_pid_tamper",
                "canonical_args_hash": canonical_effect_hash(
                    context_pid.payload["context"]
                ),
                "target_state_version": None,
            }
            context_pid.payload["effect_binding"] = changed_binding
            context_pid.payload["requested_once_capability"]["constraints"][
                "approval_binding"
            ] = dict(changed_binding)
            capture(context_pid)

            resource = deepcopy(valid)
            resource.request_id = "binding-resource-tamper"
            resource.payload["requested_once_capability"]["resource"] = (
                "filesystem:workspace:reports/other.txt"
            )
            capture(resource)

            right = deepcopy(valid)
            right.request_id = "binding-right-tamper"
            right.payload["requested_once_capability"]["rights"] = ["write"]
            capture(right)

            action = deepcopy(valid)
            action.request_id = "binding-action-tamper"
            action.payload["context"]["authority_operation"] = "filesystem.diff"
            action_binding = {
                "effect_id": "eff_action_tamper",
                "canonical_args_hash": canonical_effect_hash(
                    action.payload["context"]
                ),
                "target_state_version": None,
            }
            action.payload["effect_binding"] = action_binding
            action.payload["requested_once_capability"]["constraints"][
                "approval_binding"
            ] = dict(action_binding)
            capture(action)

            assert _assessment_for(runtime, valid.request_id).shadow_outcome == (
                "would_issue_exact_once"
            )
            assert _assessment_for(runtime, wrong_hash.request_id).shadow_outcome == (
                "require_human"
            )
            assert _assessment_for(runtime, nested.request_id).shadow_outcome == (
                "require_human"
            )
            assert _assessment_for(runtime, subject.request_id).shadow_outcome == (
                "require_human"
            )
            assert _assessment_for(runtime, context_pid.request_id).shadow_outcome == (
                "require_human"
            )
            assert _assessment_for(runtime, resource.request_id).shadow_outcome == (
                "require_human"
            )
            assert _assessment_for(runtime, right.request_id).shadow_outcome == (
                "require_human"
            )
            assert _assessment_for(runtime, action.request_id).shadow_outcome == (
                "require_human"
            )
            assert valid.status is HumanRequestStatus.PENDING
            assert tuple(runtime.capability.list_subject(pid)) == before
        finally:
            runtime.close()


def test_shadow_terminal_decision_rereads_human_revision_and_payload() -> None:
    config = AgentLibOSConfig(
        semantic=SemanticDefaults(mode="shadow", adapter="deterministic")
    )
    with TemporaryDirectory() as temp_dir:
        runtime = Runtime.open(
            "local",
            config=config,
            substrate=LocalResourceProviderSubstrate(Path(temp_dir)),
        )
        try:
            assert runtime.semantic.shutdown()
            pid = _semantic_pid(runtime)
            runtime.semantic._human_outcome_reader = lambda _request_id: "pending"
            request = _approval_request(pid, "shadow-live-revision-drift")
            runtime.human.requests.insert(request)
            queued = runtime.semantic.capture_approval(request)
            assert queued is not None
            payload = deepcopy(request.payload)
            payload["question"] = "changed after semantic capture"
            changed = runtime.human.requests.replace_current(
                request,
                payload=payload,
            )
            assert changed.status is HumanRequestStatus.PENDING
            assert changed.revision == request.revision + 1

            _drain_semantic_synchronously(runtime)
            assessment = _assessment_for(runtime, request.request_id)
            assert assessment.shadow_outcome == "require_human"
            assert "binding_current" in assessment.missing_predicates
            assert "binding_current" not in assessment.proven_predicates
            assert runtime.human.get(request.request_id) == changed
        finally:
            runtime.close()


def test_jsonrpc_and_mcp_exact_bindings_are_not_misclassified_as_malformed() -> None:
    config = AgentLibOSConfig(
        semantic=SemanticDefaults(mode="shadow", adapter="deterministic")
    )
    with TemporaryDirectory() as temp_dir:
        runtime = Runtime.open(
            "local",
            config=config,
            substrate=LocalResourceProviderSubstrate(Path(temp_dir)),
        )
        try:
            pid = runtime.process.spawn(goal="remote binding fixture")
            runtime.semantic._human_outcome_reader = lambda _request_id: "pending"
            cases = (
                (
                    "jsonrpc-valid",
                    "jsonrpc.call",
                    "jsonrpc:endpoint-a:method-a",
                    {
                        "endpoint_id": "endpoint-a",
                        "method_id": "method-a",
                        "right": "read",
                    },
                ),
                (
                    "mcp-valid",
                    "mcp.call",
                    "mcp:server-a:tool-a",
                    {
                        "server_id": "server-a",
                        "tool_id": "tool-a",
                        "right": "read",
                    },
                ),
            )
            for request_id, action, resource, identity in cases:
                context = {
                    "pid": pid,
                    "operation": action,
                    "authority_operation": action,
                    **identity,
                }
                binding = {
                    "effect_id": f"eff_{request_id}",
                    "canonical_args_hash": canonical_effect_hash(context),
                    "target_state_version": None,
                }
                now = utc_now()
                request = HumanRequest(
                    request_id=request_id,
                    pid=pid,
                    human="host",
                    payload={
                        "type": "external_operation_approval",
                        "_agent_libos_authority_request_origin": "external_operation",
                        "context": context,
                        "effect_binding": binding,
                        "requested_once_capability": {
                            "subject": pid,
                            "resource": resource,
                            "rights": ["read"],
                            "constraints": {"approval_binding": dict(binding)},
                        },
                    },
                    status=HumanRequestStatus.PENDING,
                    decision=None,
                    blocking=True,
                    created_at=now,
                    updated_at=now,
                )
                runtime.human.requests.insert(request)
                assert runtime.semantic.capture_approval(request) is not None
                record = _assessment_for(runtime, request_id)
                assert "malformed_request" not in record.reason_codes
                assert "schema_valid" in record.proven_predicates
                assert "exact_external_operation" in record.proven_predicates
                assert "binding_current" in record.proven_predicates
        finally:
            runtime.close()


def test_semantic_cursor_rejects_duplicate_keys_and_oversized_input() -> None:
    duplicate = base64.urlsafe_b64encode(
        b'{"assessment_id":"first","assessment_id":"second",'
        b'"created_at":"2026-01-01T00:00:00Z"}'
    ).decode("ascii").rstrip("=")
    with temporary_runtime() as runtime:
        with pytest.raises(ValueError, match="cursor is invalid"):
            runtime.semantic.query_assessments(after=duplicate)
        with pytest.raises(ValueError, match="cursor is invalid"):
            runtime.semantic.query_assessments(after="a" * 2049)


def test_host_tenant_bucketer_is_keyed_optional_and_failure_isolated() -> None:
    config = AgentLibOSConfig(
        semantic=SemanticDefaults(mode="shadow", adapter="deterministic")
    )
    key = b"semantic-host-only-test-key"

    def bucket(tenant: str) -> str:
        return hmac.new(key, tenant.encode("utf-8"), hashlib.sha256).hexdigest()

    for bucketer, expected in (
        (None, None),
        (bucket, bucket("tenant-a")),
    ):
        with TemporaryDirectory() as temp_dir:
            runtime = Runtime.open(
                "local",
                config=config,
                substrate=LocalResourceProviderSubstrate(Path(temp_dir)),
                semantic_tenant_bucketer=bucketer,
            )
            try:
                pid = _semantic_pid(runtime)
                request = _approval_request(pid, "tenant-bucket", tenant="tenant-a")
                job = runtime.semantic.capture_approval(request)
                assert job is not None
                assert job.bindings["tenant_bucket_sha256"] == expected
                assert job.bindings["data_labels_sha256"] != (
                    DataLabels(
                        tenant="tenant-a",
                        principal="principal-a",
                    ).labels_hash()
                )
                assert "tenant-a" not in json.dumps(to_jsonable(job), default=str)
            finally:
                runtime.close()

    with TemporaryDirectory() as temp_dir:
        runtime = Runtime.open(
            "local",
            config=config,
            substrate=LocalResourceProviderSubstrate(Path(temp_dir)),
            semantic_tenant_bucketer=lambda _tenant: "invalid",
        )
        try:
            pid = _semantic_pid(runtime)
            request = _approval_request(pid, "tenant-invalid", tenant="tenant-a")
            assert runtime.semantic.capture_approval(request) is None
            assert runtime.semantic.status()["queue"]["capture_failures"] >= 1
            assert runtime.uow.semantic.query_semantic_assessments(
                after=None,
                limit=100,
                request_id=request.request_id,
            ).records == ()
            assert runtime.uow.processes.get_process(pid).status is ProcessStatus.RUNNABLE
        finally:
            runtime.close()


def test_capture_racing_with_off_is_cancelled_before_off_returns(monkeypatch) -> None:
    config = AgentLibOSConfig(
        semantic=SemanticDefaults(mode="shadow", adapter="deterministic")
    )
    with TemporaryDirectory() as temp_dir:
        runtime = Runtime.open(
            "local",
            config=config,
            substrate=LocalResourceProviderSubstrate(Path(temp_dir)),
        )
        release = threading.Event()
        try:
            pid = _semantic_pid(runtime)
            assert runtime.semantic.shutdown()
            entered = threading.Event()
            original = runtime.semantic._capture_root_process

            def delayed_capture(*args, **kwargs):
                entered.set()
                assert release.wait(timeout=5)
                return original(*args, **kwargs)

            monkeypatch.setattr(
                runtime.semantic,
                "_capture_root_process",
                delayed_capture,
            )
            captured: list[object] = []
            failures: list[BaseException] = []

            def capture() -> None:
                try:
                    captured.append(runtime.semantic.capture_root_process(pid))
                except BaseException as exc:  # pragma: no cover - assertion aid
                    failures.append(exc)

            capture_thread = threading.Thread(target=capture)
            capture_thread.start()
            assert entered.wait(timeout=5)
            off_thread = threading.Thread(target=lambda: runtime.semantic.set_mode("off"))
            off_thread.start()
            time.sleep(0.02)
            assert off_thread.is_alive()

            release.set()
            capture_thread.join(timeout=5)
            off_thread.join(timeout=5)

            assert not capture_thread.is_alive()
            assert not off_thread.is_alive()
            assert failures == []
            assert len(captured) == 1 and captured[0] is not None
            job_id = captured[0].job_id
            persisted = runtime.uow.semantic.get_semantic_assessment_job(job_id)
            assert persisted is not None
            assert persisted.status.value == "cancelled"
            assert persisted.projection == {}
            assert runtime.semantic.mode == "off"
            assert runtime.semantic.capture_root_process(pid) is None
        finally:
            release.set()
            runtime.close()


def test_human_compatibility_observer_cannot_replace_host_semantic_capture() -> None:
    config = AgentLibOSConfig(
        semantic=SemanticDefaults(mode="shadow", adapter="deterministic")
    )
    with TemporaryDirectory() as temp_dir:
        runtime = Runtime.open(
            "local",
            config=config,
            substrate=LocalResourceProviderSubstrate(Path(temp_dir)),
        )
        try:
            pid = _semantic_pid(runtime)
            with pytest.raises(RuntimeError, match="already bound"):
                runtime.human.bind_host_request_capture(lambda _request: None)

            compatibility_seen: list[str] = []
            runtime.human.set_request_capture(
                lambda request: compatibility_seen.append(request.request_id)
            )
            first_payload = deepcopy(_approval_request(pid, "compat-first").payload)
            first_payload.pop("_agent_libos_authority_request_origin")
            first_payload.pop("effect_binding")
            first_payload["requested_once_capability"]["constraints"] = {}
            first_id = runtime.human.query_authority_request(
                pid,
                "host",
                first_payload,
                authority_origin="external_operation",
            )
            assert _assessment_for(runtime, first_id) is not None
            assert compatibility_seen == [first_id]
            runtime.human.reject(first_id, {"approved": False})

            runtime.human.set_request_capture(None)
            second_payload = deepcopy(_approval_request(pid, "compat-second").payload)
            second_payload.pop("_agent_libos_authority_request_origin")
            second_payload.pop("effect_binding")
            second_payload["requested_once_capability"]["constraints"] = {}
            second_id = runtime.human.query_authority_request(
                pid,
                "host",
                second_payload,
                authority_origin="external_operation",
            )
            assert _assessment_for(runtime, second_id) is not None
            assert compatibility_seen == [first_id]
            runtime.human.reject(second_id, {"approved": False})
        finally:
            runtime.close()


@pytest.mark.parametrize("fail_on_semantic_thread", (1, 2))
def test_semantic_thread_start_failure_degrades_capture_to_off(
    monkeypatch,
    fail_on_semantic_thread: int,
) -> None:
    original_start = threading.Thread.start
    semantic_starts = 0

    def injected_start(thread: threading.Thread) -> None:
        nonlocal semantic_starts
        if thread.name.startswith("agent-libos-semantic-"):
            semantic_starts += 1
            if semantic_starts == fail_on_semantic_thread:
                raise RuntimeError("semantic worker start unavailable")
        original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", injected_start)
    config = AgentLibOSConfig(
        semantic=SemanticDefaults(
            mode="shadow",
            adapter="deterministic",
            max_concurrency=2,
        )
    )
    with TemporaryDirectory() as temp_dir:
        runtime = Runtime.open(
            "local",
            config=config,
            substrate=LocalResourceProviderSubstrate(Path(temp_dir)),
        )
        try:
            assert runtime.semantic.mode == "off"
            assert runtime.semantic.status()["queue"]["capture_failures"] >= 1
            pid = runtime.process.spawn(goal="business launch survives Shadow failure")
            assert runtime.uow.processes.get_process(pid).status is ProcessStatus.RUNNABLE
            jobs = runtime.uow.semantic.query_semantic_assessment_jobs(
                statuses=("queued", "claimed"),
                projection_expires_before=None,
                limit=100,
            )
            assert [job for job in jobs if job.pid == pid] == []
        finally:
            runtime.close()
