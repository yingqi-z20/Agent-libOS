from __future__ import annotations

import os
import sys
import threading
import unicodedata
from dataclasses import replace
from pathlib import Path

import pytest

import agent_libos.substrate.local as local_substrate
from agent_libos import Runtime
from agent_libos.capability.manager import CapabilityManager
from agent_libos.models import CapabilityRight, DataFlowContext, DataLabels
from agent_libos.models.exceptions import CapabilityDenied
from agent_libos.substrate import (
    LocalFilesystemProvider,
    LocalResourceProviderSubstrate,
    PathState,
)


_STORED_PATH = "CaseDir/é.TXT"
_DARWIN_ALIASES = (
    pytest.param("casedir/é.txt", id="case"),
    pytest.param(
        f"CaseDir/{unicodedata.normalize('NFD', 'é')}.TXT",
        id="unicode-normalization",
    ),
)


def _darwin_alias_workspace(tmp_path: Path, alias: str) -> tuple[Path, Path]:
    if sys.platform != "darwin":
        pytest.skip("Darwin descriptor-path identity regression")
    root = tmp_path / "WorkspaceCase"
    target = root / _STORED_PATH
    target.parent.mkdir(parents=True)
    target.write_text("original", encoding="utf-8")
    alias_path = root / alias
    if not alias_path.exists() or not os.path.samefile(target, alias_path):
        pytest.skip("test volume does not map this spelling to the same entry")
    return root, target


@pytest.mark.platform_darwin
@pytest.mark.parametrize("alias", _DARWIN_ALIASES)
def test_darwin_descriptor_spelling_unifies_resource_read_and_write(
    tmp_path: Path,
    alias: str,
) -> None:
    root, target = _darwin_alias_workspace(tmp_path, alias)
    runtime = Runtime.open("local", substrate=LocalResourceProviderSubstrate(root))
    try:
        canonical_resource = runtime.filesystem.resource_for_path(_STORED_PATH)
        alias_resource = runtime.filesystem.resource_for_path(alias)
        assert alias_resource == canonical_resource

        pid = runtime.process.spawn(goal="use one filesystem identity across aliases")
        runtime.filesystem.grant_path(
            pid,
            _STORED_PATH,
            [CapabilityRight.READ, CapabilityRight.WRITE],
            issued_by="test",
        )

        assert runtime.filesystem.read_text(pid, alias).content == "original"
        written = runtime.filesystem.write_text(pid, alias, "updated")

        assert written.path == _STORED_PATH
        assert target.read_text(encoding="utf-8") == "updated"
    finally:
        runtime.close()


@pytest.mark.platform_darwin
@pytest.mark.parametrize("alias", _DARWIN_ALIASES)
def test_darwin_alias_read_observes_canonical_file_label(
    tmp_path: Path,
    alias: str,
) -> None:
    root, target = _darwin_alias_workspace(tmp_path, alias)
    runtime = Runtime.open("local", substrate=LocalResourceProviderSubstrate(root))
    try:
        pid = runtime.process.spawn(goal="preserve a secret filesystem label")
        runtime.filesystem.grant_directory(
            pid,
            ".",
            [CapabilityRight.READ],
            issued_by="test",
        )
        runtime.data_flow.bind_written_file(
            pid=pid,
            normalized_path=_STORED_PATH,
            content=target.read_bytes(),
            context=DataFlowContext(labels=DataLabels(sensitivity="secret")),
        )

        with runtime.data_flow.activate(DataFlowContext()):
            assert runtime.filesystem.read_text(pid, alias).content == "original"
            assert (
                runtime.data_flow.current_context().labels.sensitivity.value
                == "secret"
            )
    finally:
        runtime.close()


@pytest.mark.platform_darwin
@pytest.mark.parametrize("alias", _DARWIN_ALIASES)
def test_darwin_alias_cannot_bypass_exact_deny(
    tmp_path: Path,
    alias: str,
) -> None:
    root, _target = _darwin_alias_workspace(tmp_path, alias)
    runtime = Runtime.open("local", substrate=LocalResourceProviderSubstrate(root))
    try:
        pid = runtime.process.spawn(goal="enforce an exact filesystem deny")
        runtime.filesystem.grant_directory(
            pid,
            ".",
            [CapabilityRight.READ],
            issued_by="test",
        )
        runtime.capability.set_permission_policy(
            subject=pid,
            resource=runtime.filesystem.resource_for_path(_STORED_PATH),
            rights=[CapabilityRight.READ],
            policy=CapabilityManager.ALWAYS_DENY,
            issued_by="test",
        )

        with pytest.raises(CapabilityDenied):
            runtime.filesystem.read_text(pid, alias)
    finally:
        runtime.close()


@pytest.mark.platform_darwin
def test_darwin_root_alias_uses_descriptor_canonical_spelling(tmp_path: Path) -> None:
    if sys.platform != "darwin":
        pytest.skip("Darwin descriptor-path identity regression")
    root = tmp_path / "WorkspaceCase"
    root.mkdir()
    alias = root.with_name("workspacecase")
    if not alias.exists() or not os.path.samefile(root, alias):
        pytest.skip("test volume is case-sensitive")

    provider = LocalFilesystemProvider(alias)
    substrate = LocalResourceProviderSubstrate(alias)

    assert provider.root.name == "WorkspaceCase"
    assert substrate.workspace_root == provider.root
    assert substrate.filesystem.root == provider.root


@pytest.mark.platform_darwin
def test_darwin_fgetpath_failure_has_no_lexical_alias_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sys.platform != "darwin":
        pytest.skip("Darwin descriptor-path identity regression")
    root = tmp_path / "workspace"
    root.mkdir()
    import fcntl

    provider = LocalFilesystemProvider(root)

    def fail_fgetpath(*_args: object, **_kwargs: object) -> bytes:
        raise OSError("simulated F_GETPATH failure")

    monkeypatch.setattr(fcntl, "fcntl", fail_fgetpath)
    with pytest.raises(CapabilityDenied, match="canonicalize.*path identity"):
        provider.resolve("missing.txt")
    with pytest.raises(CapabilityDenied, match="canonicalize.*root identity"):
        LocalFilesystemProvider(root)


@pytest.mark.platform_darwin
@pytest.mark.parametrize(
    ("denied_path", "alias"),
    (
        pytest.param("Future/Report.TXT", "future/report.txt", id="case"),
        pytest.param(
            "Future/é.txt",
            f"Future/{unicodedata.normalize('NFD', 'é')}.txt",
            id="unicode-normalization",
        ),
    ),
)
def test_darwin_missing_alias_cannot_bypass_exact_deny(
    tmp_path: Path,
    denied_path: str,
    alias: str,
) -> None:
    if sys.platform != "darwin":
        pytest.skip("Darwin future-entry identity regression")
    root = tmp_path / "workspace"
    root.mkdir()
    runtime = Runtime.open("local", substrate=LocalResourceProviderSubstrate(root))
    try:
        denied_resource = runtime.filesystem.resource_for_path(denied_path)
        assert runtime.filesystem.resource_for_path(alias) == denied_resource
        pid = runtime.process.spawn(goal="enforce a deny before a file exists")
        runtime.filesystem.grant_directory(
            pid,
            ".",
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        runtime.capability.set_permission_policy(
            subject=pid,
            resource=denied_resource,
            rights=[CapabilityRight.WRITE],
            policy=CapabilityManager.ALWAYS_DENY,
            issued_by="test",
        )

        with pytest.raises(CapabilityDenied):
            runtime.filesystem.write_text(pid, alias, "blocked")

        assert list(root.iterdir()) == []
    finally:
        runtime.close()


@pytest.mark.platform_darwin
def test_darwin_missing_case_and_unicode_aliases_share_creation_lock(
    tmp_path: Path,
) -> None:
    if sys.platform != "darwin":
        pytest.skip("Darwin future-entry identity regression")

    class BlockingWriteProvider(LocalFilesystemProvider):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.first_entered = threading.Event()
            self.release_first = threading.Event()
            self.second_entered = threading.Event()
            self._entry_lock = threading.Lock()
            self._entries = 0

        def _before_path_sink(self, operation: str, target: Path) -> None:
            if operation != "write_text":
                return
            with self._entry_lock:
                self._entries += 1
                entry = self._entries
            if entry == 1:
                self.first_entered.set()
                assert self.release_first.wait(timeout=5)
            else:
                self.second_entered.set()

    root = tmp_path / "workspace"
    root.mkdir()
    provider = BlockingWriteProvider(root)
    first = provider.resolve("Future/é.TXT")
    second = provider.resolve(
        f"future/{unicodedata.normalize('NFD', 'é')}.txt"
    )
    assert first.relative == second.relative == "future/é.txt"
    errors: list[BaseException] = []

    def write(path: object, content: str) -> None:
        try:
            provider.write_text(path, content, "utf-8")  # type: ignore[arg-type]
        except BaseException as exc:
            errors.append(exc)

    first_thread = threading.Thread(target=write, args=(first, "first"), daemon=True)
    second_thread = threading.Thread(target=write, args=(second, "second"), daemon=True)
    first_thread.start()
    assert provider.first_entered.wait(timeout=5)
    second_thread.start()
    assert not provider.second_entered.wait(timeout=0.2)
    provider.release_first.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert errors == []
    assert provider.second_entered.is_set()
    assert (root / "future" / "é.txt").read_text(encoding="utf-8") == "second"


@pytest.mark.platform_darwin
def test_darwin_case_sensitive_policy_preserves_distinct_future_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sys.platform != "darwin":
        pytest.skip("Darwin future-entry identity regression")
    root = tmp_path / "workspace"
    root.mkdir()
    provider = LocalFilesystemProvider(root)
    assert provider._darwin_volume_policy is not None
    selected_policy = replace(
        provider._darwin_volume_policy,
        case_sensitive=True,
    )
    provider._darwin_volume_policy = selected_policy
    monkeypatch.setattr(
        local_substrate,
        "_darwin_volume_identity_policy",
        lambda _path: selected_policy,
    )

    upper = provider.resolve("Future/Report.txt")
    lower = provider.resolve("future/report.txt")

    assert upper.relative == "Future/Report.txt"
    assert lower.relative == "future/report.txt"
    assert upper.relative != lower.relative


@pytest.mark.platform_darwin
@pytest.mark.parametrize(
    "component",
    (
        pytest.param("ß.txt", id="sharp-s"),
        pytest.param("İ.txt", id="dotted-capital-i"),
        pytest.param("ﬃ.txt", id="ligature"),
    ),
)
def test_darwin_ambiguous_non_ascii_future_casefold_fails_closed(
    tmp_path: Path,
    component: str,
) -> None:
    if sys.platform != "darwin":
        pytest.skip("Darwin future-entry identity regression")
    root = tmp_path / "workspace"
    root.mkdir()
    provider = LocalFilesystemProvider(root)
    policy = provider._darwin_volume_policy
    if policy is None or policy.case_sensitive:
        pytest.skip("test volume is not known case-insensitive")

    with pytest.raises(CapabilityDenied, match="non-ASCII case-insensitive"):
        provider.resolve(f"future/{component}")


@pytest.mark.platform_darwin
def test_darwin_unknown_volume_policy_allows_existing_but_denies_future_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sys.platform != "darwin":
        pytest.skip("Darwin future-entry identity regression")
    root = tmp_path / "workspace"
    root.mkdir()
    existing = root / "existing.txt"
    existing.write_text("existing", encoding="utf-8")

    def unavailable(_path: Path) -> object:
        raise CapabilityDenied("simulated unavailable volume capabilities")

    monkeypatch.setattr(
        local_substrate,
        "_darwin_volume_identity_policy",
        unavailable,
    )
    provider = LocalFilesystemProvider(root)

    assert provider.read_bytes(provider.resolve("existing.txt")) == b"existing"
    with pytest.raises(CapabilityDenied, match="unavailable volume capabilities"):
        provider.resolve("future.txt")


@pytest.mark.platform_linux
def test_linux_case_and_unicode_distinct_files_keep_distinct_identities(
    tmp_path: Path,
) -> None:
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux case-sensitive filesystem control")
    root = tmp_path / "workspace"
    root.mkdir()
    names = (
        "Case.txt",
        "case.txt",
        "é.txt",
        f"{unicodedata.normalize('NFD', 'é')}.txt",
    )
    for index, name in enumerate(names):
        (root / name).write_text(str(index), encoding="utf-8")
    if len({os.stat(root / name).st_ino for name in names}) != len(names):
        pytest.skip("test filesystem aliases one of the Linux control names")

    provider = LocalFilesystemProvider(root)
    resolved = [provider.resolve(name) for name in names]

    assert [path.relative for path in resolved] == list(names)
    assert [provider.read_bytes(path) for path in resolved] == [
        str(index).encode("utf-8") for index in range(len(names))
    ]


def test_filesystem_state_preserves_file_directory_root_and_missing_semantics(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    regular_file = root / "file.txt"
    regular_file.write_bytes(b"inside")
    (root / "directory").mkdir()
    provider = LocalFilesystemProvider(root)

    file_state = provider.state(provider.resolve("file.txt"))
    directory_state = provider.state(provider.resolve("directory"))
    root_state = provider.state(provider.resolve("."))
    missing_state = provider.state(provider.resolve("missing/child"))

    assert file_state.exists is True
    assert file_state.kind == "file"
    assert file_state.size_bytes == len(b"inside")
    assert file_state.modified_at is not None
    assert directory_state.exists is True
    assert directory_state.kind == "directory"
    assert directory_state.size_bytes is None
    assert root_state.exists is True
    assert root_state.kind == "directory"
    assert missing_state.exists is False
    assert missing_state.kind == "missing"
    assert missing_state.size_bytes is None
    assert missing_state.modified_at is None


def test_filesystem_state_rejects_final_symlink_swap_before_metadata_projection(
    tmp_path: Path,
) -> None:
    if os.name == "nt":
        pytest.skip("deterministic symlink swap requires unprivileged symlink creation")

    class StateSwapProvider(LocalFilesystemProvider):
        def __init__(self, root: Path, outside: Path) -> None:
            super().__init__(root)
            self.outside = outside
            self.swapped = False
            self.projected_state = False

        def _before_path_sink(self, operation: str, target: Path) -> None:
            if operation != "state" or self.swapped:
                return
            self.swapped = True
            target.unlink()
            target.symlink_to(self.outside)

        def _path_state_from_stat(  # type: ignore[override]
            self,
            observed: os.stat_result,
        ) -> PathState:
            self.projected_state = True
            return super()._path_state_from_stat(observed)

    root = tmp_path / "workspace"
    root.mkdir()
    victim = root / "victim.txt"
    victim.write_bytes(b"inside")
    outside = tmp_path / "outside-secret.txt"
    outside_marker = b"outside-secret-metadata-marker"
    outside.write_bytes(outside_marker)
    provider = StateSwapProvider(root, outside)

    with pytest.raises(CapabilityDenied) as exc_info:
        provider.state(provider.resolve("victim.txt"))

    assert provider.swapped is True
    assert provider.projected_state is False
    error = str(exc_info.value)
    assert outside_marker.decode("ascii") not in error
    assert str(outside) not in error
    assert outside.read_bytes() == outside_marker
