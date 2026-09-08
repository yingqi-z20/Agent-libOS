from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from agent_libos.models.exceptions import ValidationError
from agent_libos.models.snapshot import LocalProviderContinuationReference
from agent_libos.runtime.snapshots import SnapshotCodec, SnapshotIdentityMap, SnapshotRemapper
from tests.unit.test_snapshot_types import _snapshot


def _reference(**changes: object) -> dict[str, object]:
    return {
        "pid": "pid_1",
        "marker_call_id": "llmcontinuation_1",
        "marker_sha256": "a" * 64,
        "source_call_id": "llmcall_1",
        "source_sha256": "b" * 64,
        "source_pid": "pid_1",
        "profile_identity_sha256": "c" * 64,
        "context_generation": "initial",
        "payload_sha256": "d" * 64,
        **changes,
    }


def _with_reference(**changes: object) -> dict:
    return {
        **_snapshot(),
        "provider_continuation_refs": {"pid_1": _reference(**changes)},
    }


def test_snapshot_provider_continuation_reference_is_optional_and_lossless() -> None:
    legacy = _snapshot()
    decoded = SnapshotCodec.decode_mapping(legacy)
    assert decoded.provider_continuation_refs == {}
    assert SnapshotCodec.encode_mapping(decoded) == legacy

    raw = _with_reference()
    typed = SnapshotCodec.decode_mapping(raw)
    reference = typed.provider_continuation_refs["pid_1"]
    assert isinstance(reference, LocalProviderContinuationReference)
    assert reference.to_mapping() == _reference()
    assert SnapshotCodec.encode_mapping(typed) == raw
    assert SnapshotCodec.loads(SnapshotCodec.dumps(typed)) == typed
    assert typed.header.schema_version == decoded.header.schema_version
    with pytest.raises(FrozenInstanceError):
        reference.pid = "pid_2"


@pytest.mark.parametrize("field", sorted(LocalProviderContinuationReference.FIELDS))
@pytest.mark.parametrize("value", [None, 1, "", " leading", "trailing "])
def test_provider_continuation_reference_requires_canonical_nonempty_text(
    field: str, value: object,
) -> None:
    with pytest.raises(ValidationError, match="provider continuation reference"):
        LocalProviderContinuationReference.from_mapping(_reference(**{field: value}))


@pytest.mark.parametrize("field", [
    "marker_sha256", "source_sha256", "profile_identity_sha256", "payload_sha256",
])
@pytest.mark.parametrize("value", ["A" * 64, "a" * 63, "a" * 65, "g" * 64])
def test_provider_continuation_reference_requires_lowercase_sha256(
    field: str, value: str,
) -> None:
    with pytest.raises(ValidationError, match="SHA-256 digest"):
        LocalProviderContinuationReference.from_mapping(_reference(**{field: value}))


@pytest.mark.parametrize("field", sorted(LocalProviderContinuationReference.FIELDS))
def test_provider_continuation_reference_rejects_missing_fields(field: str) -> None:
    raw = _reference()
    raw.pop(field)
    with pytest.raises(ValidationError, match="fields are not canonical"):
        LocalProviderContinuationReference.from_mapping(raw)


@pytest.mark.parametrize("extra", ["provider_payload", "container_id", "run_id"])
def test_provider_continuation_reference_rejects_additional_payload_fields(extra: str) -> None:
    with pytest.raises(ValidationError, match="fields are not canonical"):
        SnapshotCodec.decode_mapping(_with_reference(**{extra: "must-not-be-exported"}))


@pytest.mark.parametrize("mapping_pid, reference_pid", [
    ("pid_other", "pid_other"), ("pid_1", "pid_other"), ("pid_other", "pid_1"),
])
def test_snapshot_provider_continuation_reference_rejects_process_scope_mismatch(
    mapping_pid: str, reference_pid: str,
) -> None:
    raw = {
        **_snapshot(),
        "provider_continuation_refs": {mapping_pid: _reference(pid=reference_pid)},
    }
    with pytest.raises(ValidationError, match="outside its process scope"):
        SnapshotCodec.decode_mapping(raw)


def test_snapshot_provider_continuation_reference_preserves_original_source_process() -> None:
    raw = _with_reference(source_pid="original_process_before_fork")
    typed = SnapshotCodec.decode_mapping(raw)
    assert SnapshotCodec.encode_mapping(typed) == raw


def test_snapshot_provider_continuation_reference_rejects_task_run_owned_process() -> None:
    raw = _with_reference()
    raw["rows"]["processes"][0].update({
        "task_run_id": "taskrun_1", "task_run_epoch": 1, "task_run_role": "root",
    })
    with pytest.raises(ValidationError, match="must not belong to a TaskRun"):
        SnapshotCodec.decode_mapping(raw)


def test_direct_snapshot_provider_continuation_reference_requires_typed_value() -> None:
    typed = SnapshotCodec.decode_mapping(_snapshot())
    with pytest.raises(ValidationError, match="reference must be typed"):
        replace(typed, provider_continuation_refs={"pid_1": _reference()})


@pytest.mark.parametrize("identities", [
    SnapshotIdentityMap(),
    SnapshotIdentityMap(pids={"pid_1": "pid_1"}, objects={"obj_1": "obj_1"}),
])
def test_snapshot_remapper_preserves_continuation_reference_with_unchanged_identity(
    identities: SnapshotIdentityMap,
) -> None:
    typed = SnapshotCodec.decode_mapping(_with_reference())
    remapped = SnapshotRemapper.remap(typed, identities)
    assert remapped.provider_continuation_refs == typed.provider_continuation_refs
    assert remapped.provider_continuation_refs is not typed.provider_continuation_refs


@pytest.mark.parametrize("identities", [
    SnapshotIdentityMap(pids={"pid_1": "pid_2"}),
    SnapshotIdentityMap(objects={"obj_1": "obj_2"}),
])
def test_snapshot_remapper_requires_authorized_continuation_rebind(
    identities: SnapshotIdentityMap,
) -> None:
    typed = SnapshotCodec.decode_mapping(_with_reference())
    with pytest.raises(ValidationError, match="authorized local rebind"):
        SnapshotRemapper.remap(typed, identities)
