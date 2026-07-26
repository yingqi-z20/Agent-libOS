from __future__ import annotations

import inspect
import threading
from types import MappingProxyType

import pytest

from agent_libos import Runtime
from agent_libos.models import (
    CapabilityRight,
    DataFlowContext,
    MemoryView,
    MergePolicy,
    ObjectOwnerKind,
    ObjectType,
    Provenance,
    RelationType,
    ViewMode,
)
from agent_libos.models.exceptions import CapabilityDenied, NotFound, ValidationError
from agent_libos.ports import DataFlowPort


def test_data_flow_port_exclusion_call_shape_and_semantics_match_runtime() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="exclude Object provenance")
        first = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {"first": True},
        )
        second = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {"second": True},
        )
        first_obj = runtime.store.get_object(first.oid)
        second_obj = runtime.store.get_object(second.oid)
        assert first_obj is not None and second_obj is not None
        context = DataFlowContext.aggregate(
            (
                runtime.data_flow.context_from_object_snapshot(first_obj),
                runtime.data_flow.context_from_object_snapshot(second_obj),
            )
        )

        inspect.signature(DataFlowPort.provenance_sources).bind(
            None,
            context,
            exclude_oids=[first.oid],
        )
        inspect.signature(runtime.data_flow.provenance_sources).bind(
            context,
            exclude_oids=[first.oid],
        )
        parent_oids, durable_refs = runtime.data_flow.provenance_sources(
            context,
            exclude_oids=[first.oid],
        )

        assert parent_oids == (second.oid,)
        assert durable_refs == ()
    finally:
        runtime.close()


def test_named_read_and_append_hide_existing_entries_before_payload_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        owner = runtime.process.spawn(goal="own hidden Objects")
        observer = runtime.process.spawn(goal="probe hidden Objects")
        namespace = runtime.memory.create_namespace(owner, "hidden-name-probes")
        runtime.capability.grant(
            observer,
            f"object_namespace:{namespace.namespace}",
            [CapabilityRight.READ],
            issued_by="test.host",
        )

        with pytest.raises(NotFound) as missing_read:
            runtime.memory.get_object_by_name(
                observer,
                "read-probe",
                namespace=namespace.namespace,
            )
        with pytest.raises(NotFound) as missing_append:
            runtime.memory.append_object_by_name(
                observer,
                "append-probe",
                {"probe": True},
                namespace=namespace.namespace,
            )

        runtime.memory.create_object(
            owner,
            ObjectType.EVIDENCE,
            {"secret": "must not load"},
            name="read-probe",
            namespace=namespace.namespace,
        )
        runtime.memory.create_object(
            owner,
            ObjectType.OBSERVATION,
            {"entries": []},
            name="append-probe",
            namespace=namespace.namespace,
            immutable=False,
        )

        def fail_payload_load(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("hidden Object payload was loaded before exact rights")

        monkeypatch.setattr(runtime.store, "get_object", fail_payload_load)
        with pytest.raises(NotFound) as hidden_read:
            runtime.memory.get_object_by_name(
                observer,
                "read-probe",
                namespace=namespace.namespace,
            )
        with pytest.raises(NotFound) as hidden_append:
            runtime.memory.append_object_by_name(
                observer,
                "append-probe",
                {"probe": True},
                namespace=namespace.namespace,
            )

        assert str(hidden_read.value) == str(missing_read.value)
        assert str(hidden_append.value) == str(missing_append.value)
    finally:
        runtime.close()


def test_llm_parent_validation_authorizes_before_payload_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        owner = runtime.process.spawn(goal="own provenance parent")
        creator = runtime.process.spawn(goal="create derived Object")
        parent = runtime.memory.create_object(
            owner,
            ObjectType.EVIDENCE,
            {"secret": "parent payload"},
        )
        child_name = "derived-parent-probe"

        with pytest.raises(CapabilityDenied) as missing:
            runtime.memory.create_object(
                creator,
                ObjectType.ARTIFACT,
                {"child": True},
                name=child_name,
                provenance=Provenance(
                    created_from_action="llm.tool",
                    parent_oids=["obj_missing_parent_probe"],
                ),
            )

        def fail_payload_load(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("parent payload was loaded before exact READ")

        monkeypatch.setattr(runtime.store, "get_object", fail_payload_load)
        with pytest.raises(CapabilityDenied) as hidden:
            runtime.memory.create_object(
                creator,
                ObjectType.ARTIFACT,
                {"child": True},
                name=child_name,
                provenance=Provenance(
                    created_from_action="llm.tool",
                    parent_oids=[parent.oid],
                ),
            )

        assert str(hidden.value) == str(missing.value)
    finally:
        runtime.close()


def test_duplicate_view_roots_are_deduplicated_for_fork_and_materialization() -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(goal="deduplicate MemoryView roots")
        child = runtime.process.spawn(goal="receive one MemoryView root")
        handle = runtime.memory.create_object(
            parent,
            ObjectType.EVIDENCE,
            {"marker": "DUPLICATE_ROOT_SENTINEL"},
        )

        view = runtime.memory.create_view(parent, [handle, handle])
        assert [root.oid for root in view.roots] == [handle.oid]
        duplicated_view = MemoryView(
            view_id="view_duplicate_external",
            owner_pid=parent,
            roots=[handle, handle],
            filters=[],
            rights_policy="attenuate",
            created_from=None,
            mode=ViewMode.READ_ONLY,
        )
        materialized = runtime.memory.materialize_context(
            parent,
            duplicated_view,
            budget_tokens=10_000,
        )
        forked = runtime.memory.fork_view(parent, child, duplicated_view)

        assert materialized.text.count("DUPLICATE_ROOT_SENTINEL") == 1
        assert materialized.object_refs == [handle.oid]
        assert len(materialized.object_manifest) == 1
        assert materialized.object_manifest[0]["disposition"] == "included"
        assert [root.oid for root in forked.roots] == [handle.oid]
    finally:
        runtime.close()


def test_merge_policy_include_updated_controls_inherited_child_roots() -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(goal="merge inherited Object")
        child = runtime.process.spawn(goal="hold inherited Object")
        handle = runtime.memory.create_object(
            parent,
            ObjectType.EVIDENCE,
            {"updated": True},
        )
        parent_view = runtime.memory.create_view(parent, [handle])
        child_view = runtime.memory.fork_view(parent, child, parent_view)

        excluded = runtime.memory.merge_view(
            parent,
            child_view,
            MergePolicy(include_child_created=False, include_updated=False),
        )
        included = runtime.memory.merge_view(
            parent,
            child_view,
            MergePolicy(include_child_created=False, include_updated=True),
        )

        assert excluded.merged_oids == []
        assert excluded.merged_handles == []
        assert included.merged_oids == [handle.oid]
        assert [merged.oid for merged in included.merged_handles] == [handle.oid]
    finally:
        runtime.close()


def test_link_metadata_is_strict_bounded_json_for_public_and_trusted_paths() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="validate Object link metadata")
        source = runtime.memory.create_object(pid, ObjectType.ARTIFACT, {"src": True})
        destination = runtime.memory.create_object(pid, ObjectType.EVIDENCE, {"dst": True})
        cycle: dict[str, object] = {}
        cycle["self"] = cycle
        deep: dict[str, object] = {}
        cursor = deep
        for _ in range(70):
            nested: dict[str, object] = {}
            cursor["nested"] = nested
            cursor = nested
        invalid_metadata: tuple[object, ...] = (
            ["not", "an", "object"],
            {"value": float("nan")},
            cycle,
            deep,
            {"padding": "x" * (runtime.config.memory.metadata_max_bytes + 1)},
        )

        for metadata in invalid_metadata:
            with pytest.raises(ValidationError):
                runtime.memory.link_objects(
                    pid,
                    source,
                    RelationType.REFERENCES,
                    destination,
                    metadata,  # type: ignore[arg-type]
                )
            with pytest.raises(ValidationError):
                runtime.memory.link_objects_trusted(
                    "test.host",
                    source.oid,
                    RelationType.REFERENCES,
                    destination.oid,
                    metadata,  # type: ignore[arg-type]
                    reason="link metadata regression",
                )

        assert runtime.store.list_links(src=source.oid) == []
        metadata = MappingProxyType(
            {"provider": MappingProxyType({"resource": "remote-1"})}
        )
        runtime.memory.link_objects(
            pid,
            source,
            RelationType.REFERENCES,
            destination,
            metadata,
        )
        links = runtime.store.list_links(src=source.oid)
        assert len(links) == 1
        assert links[0].metadata == {"provider": {"resource": "remote-1"}}
    finally:
        runtime.close()


def test_object_task_publication_and_owner_release_share_one_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="publish pinned task")
        runtime.capability.grant(
            pid,
            "process:spawn",
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )
        owner = runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {"owner": True},
            immutable=False,
        )
        publication_entered = threading.Event()
        allow_publication = threading.Event()
        release_done = threading.Event()
        original_insert = runtime.object_tasks._records.insert_object_task

        def pause_publication(task: object) -> None:
            publication_entered.set()
            assert allow_publication.wait(timeout=3)
            original_insert(task)

        monkeypatch.setattr(
            runtime.object_tasks._records,
            "insert_object_task",
            pause_publication,
        )
        monkeypatch.setattr(
            runtime.object_tasks,
            "_schedule_task_locked",
            lambda _task_id: None,
        )
        started: list[object] = []
        released: list[list[str]] = []
        errors: list[BaseException] = []

        def start_task() -> None:
            try:
                started.append(
                    runtime.object_tasks.start(
                        pid,
                        owner,
                        "get_working_directory",
                        {},
                    )
                )
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        def release_owner() -> None:
            try:
                released.append(
                    runtime.memory.release_owner(
                        ObjectOwnerKind.PROCESS,
                        pid,
                        preserve_oids={runtime.process.get(pid).goal_oid},
                    )
                )
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)
            finally:
                release_done.set()

        start_thread = threading.Thread(target=start_task)
        start_thread.start()
        assert publication_entered.wait(timeout=3)
        release_thread = threading.Thread(target=release_owner)
        release_thread.start()
        assert not release_done.wait(timeout=0.1)
        allow_publication.set()
        start_thread.join(timeout=5)
        release_thread.join(timeout=5)

        assert not start_thread.is_alive()
        assert not release_thread.is_alive()
        assert errors == []
        assert len(started) == 1
        assert released == [[]]
        assert runtime.store.get_object(owner.oid) is not None
        assert runtime.object_tasks.has_published_active_for_owner(owner.oid)
    finally:
        runtime.close()
