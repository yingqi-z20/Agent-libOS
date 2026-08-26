from __future__ import annotations

import ctypes
import hashlib
import os
import struct
import subprocess
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
_WINDOWS_STORED_PATH = "CaseDir/Évidence.TXT"
_WINDOWS_ALIAS_PATH = "casedir/évidence.txt"


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


@pytest.mark.parametrize(
    "path",
    (
        pytest.param("report.txt:secret", id="alternate-data-stream"),
        pytest.param("CON.txt", id="device-with-extension"),
        pytest.param("CON .txt", id="device-with-space-before-extension"),
        pytest.param("COM¹.log", id="superscript-device-with-extension"),
        pytest.param("aux", id="device"),
        pytest.param("folder.", id="trailing-dot"),
        pytest.param("folder ", id="trailing-space"),
        pytest.param("bad<name", id="illegal-character"),
        pytest.param("control\x01", id="control-character"),
        pytest.param(r"\\?\C:\workspace\report.txt", id="device-path"),
        pytest.param(r"C:relative.txt", id="drive-relative"),
        pytest.param(r"\root-relative.txt", id="root-relative"),
    ),
)
def test_windows_ambiguous_path_syntax_fails_before_identity_derivation(
    path: str,
) -> None:
    with pytest.raises(CapabilityDenied):
        local_substrate._validate_windows_path_syntax(path)


def test_windows_future_component_policy_is_deterministic_and_fails_closed() -> None:
    assert LocalFilesystemProvider._windows_future_component(
        "Report.TXT",
        case_sensitive=False,
    ) == "report.txt"
    assert LocalFilesystemProvider._windows_future_component(
        "Report.TXT",
        case_sensitive=True,
    ) == "Report.TXT"
    with pytest.raises(CapabilityDenied, match="non-ASCII future path"):
        LocalFilesystemProvider._windows_future_component(
            "Évidence.txt",
            case_sensitive=False,
        )
    with pytest.raises(CapabilityDenied, match="DOS short name"):
        LocalFilesystemProvider._windows_future_component(
            "REPORT~1.TXT",
            case_sensitive=False,
        )


def test_windows_sink_identity_comparison_requires_canonical_spelling() -> None:
    canonical = Path(r"C:\Workspace\CaseDir\Evidence.txt")
    case_alias = Path(r"c:\workspace\casedir\evidence.TXT")

    assert LocalFilesystemProvider._windows_paths_match_exactly(
        canonical,
        canonical,
    )
    assert not LocalFilesystemProvider._windows_paths_match_exactly(
        canonical,
        case_alias,
    )


def _windows_alias_workspace(tmp_path: Path) -> tuple[Path, Path]:
    if os.name != "nt":
        pytest.skip("native Windows filesystem identity regression")
    root = tmp_path / "WorkspaceIdentity"
    target = root / _WINDOWS_STORED_PATH
    target.parent.mkdir(parents=True)
    target.write_text("original", encoding="utf-8")
    alias = root / _WINDOWS_ALIAS_PATH
    if not alias.exists() or not os.path.samefile(target, alias):
        pytest.fail("Windows test volume is not case-insensitive")
    return root, target


class _WindowsCaseSensitiveInfo(ctypes.Structure):
    _fields_ = [("Flags", ctypes.c_uint32)]


def _windows_enable_case_sensitive_directory(path: Path) -> None:
    if os.name != "nt":
        raise OSError("native Windows case-sensitive directory test")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create_file.restype = ctypes.c_void_p
    set_information = kernel32.SetFileInformationByHandle
    set_information.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    set_information.restype = ctypes.c_int
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle.restype = ctypes.c_int
    handle = create_file(
        os.fspath(path),
        0x00000100,  # FILE_WRITE_ATTRIBUTES
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        3,  # OPEN_EXISTING
        0x02000000,  # FILE_FLAG_BACKUP_SEMANTICS
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if not handle or handle == invalid:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        information = _WindowsCaseSensitiveInfo(Flags=1)
        if not set_information(
            handle,
            23,  # FileCaseSensitiveInfo
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        close_handle(handle)


def _windows_replace_file(target: Path, backup: Path, content: str) -> None:
    target.rename(backup)
    target.write_text(content, encoding="utf-8")


def _windows_open_directory_reparse_writer(path: Path) -> tuple[object, object]:
    if os.name != "nt":
        raise OSError("native Windows directory reparse test")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create_file.restype = ctypes.c_void_p
    handle = create_file(
        os.fspath(path),
        0x40000000,  # GENERIC_WRITE
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        3,  # OPEN_EXISTING
        0x02000000 | 0x00200000,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if not handle or handle == invalid:
        raise ctypes.WinError(ctypes.get_last_error())
    return kernel32, handle


def _windows_set_junction_on_handle(
    kernel32: object,
    handle: object,
    target: Path,
) -> None:
    device_io_control = kernel32.DeviceIoControl
    device_io_control.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
    ]
    device_io_control.restype = ctypes.c_int
    substitute = ("\\??\\" + os.fspath(target)).encode("utf-16-le")
    display = os.fspath(target).encode("utf-16-le")
    path_buffer = substitute + b"\x00\x00" + display + b"\x00\x00"
    payload = struct.pack(
        "<IHHHHHH",
        0xA0000003,  # IO_REPARSE_TAG_MOUNT_POINT
        8 + len(path_buffer),
        0,
        0,
        len(substitute),
        len(substitute) + 2,
        len(display),
    ) + path_buffer
    buffer = ctypes.create_string_buffer(payload)
    returned = ctypes.c_uint32()
    if not device_io_control(
        handle,
        0x000900A4,  # FSCTL_SET_REPARSE_POINT
        buffer,
        len(payload),
        None,
        0,
        ctypes.byref(returned),
        None,
    ):
        raise ctypes.WinError(ctypes.get_last_error())


def _windows_mutate_directory_to_junction(path: Path, target: Path) -> None:
    kernel32, handle = _windows_open_directory_reparse_writer(path)
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle.restype = ctypes.c_int
    try:
        _windows_set_junction_on_handle(kernel32, handle, target)
    finally:
        close_handle(handle)


@pytest.mark.platform_windows
def test_windows_provider_rejects_ambiguous_spelling_before_identity_or_effect(
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("native Windows path syntax regression")
    root = tmp_path / "workspace"
    root.mkdir()
    provider = LocalFilesystemProvider(root)

    for invalid in (
        "report.txt:secret",
        "CON.txt",
        "trailing.",
        "trailing ",
        "bad<name",
        "control\x01",
        r"\\?\C:\workspace\report.txt",
    ):
        with pytest.raises(CapabilityDenied):
            provider.resolve(invalid)

    assert list(root.iterdir()) == []


@pytest.mark.platform_windows
def test_windows_existing_alias_unifies_resource_and_secret_label(
    tmp_path: Path,
) -> None:
    root, target = _windows_alias_workspace(tmp_path)
    runtime = Runtime.open("local", substrate=LocalResourceProviderSubstrate(root))
    try:
        canonical_resource = runtime.filesystem.resource_for_path(
            _WINDOWS_STORED_PATH
        )
        alias_resource = runtime.filesystem.resource_for_path(_WINDOWS_ALIAS_PATH)
        assert alias_resource == canonical_resource

        pid = runtime.process.spawn(goal="preserve Windows file-label identity")
        runtime.filesystem.grant_directory(
            pid,
            ".",
            [CapabilityRight.READ],
            issued_by="test",
        )
        runtime.data_flow.bind_written_file(
            pid=pid,
            normalized_path=_WINDOWS_STORED_PATH,
            content=target.read_bytes(),
            context=DataFlowContext(labels=DataLabels(sensitivity="secret")),
        )

        with runtime.data_flow.activate(DataFlowContext()):
            assert (
                runtime.filesystem.read_text(pid, _WINDOWS_ALIAS_PATH).content
                == "original"
            )
            assert (
                runtime.data_flow.current_context().labels.sensitivity.value
                == "secret"
            )
    finally:
        runtime.close()


@pytest.mark.platform_windows
def test_windows_canonical_identity_reaches_result_audit_and_effect_evidence(
    tmp_path: Path,
) -> None:
    root, target = _windows_alias_workspace(tmp_path)
    runtime = Runtime.open("local", substrate=LocalResourceProviderSubstrate(root))
    try:
        pid = runtime.process.spawn(goal="record one Windows path identity")
        runtime.filesystem.grant_directory(
            pid,
            ".",
            [CapabilityRight.WRITE],
            issued_by="test",
        )
        canonical_resource = runtime.filesystem.resource_for_path(
            _WINDOWS_STORED_PATH
        )

        result = runtime.filesystem.write_text(
            pid,
            _WINDOWS_ALIAS_PATH,
            "updated",
        )

        assert result.path == _WINDOWS_STORED_PATH
        assert target.read_text(encoding="utf-8") == "updated"
        effects = [
            effect
            for effect in runtime.store.list_external_effects(pid=pid)
            if effect.provider == "filesystem" and effect.operation == "write_text"
        ]
        assert len(effects) == 1
        assert effects[0].provider_metadata["path"] == _WINDOWS_STORED_PATH
        assert canonical_resource in {
            record.target
            for record in runtime.store.list_audit()
            if record.actor == pid
        }
    finally:
        runtime.close()


@pytest.mark.platform_windows
def test_windows_existing_alias_cannot_bypass_exact_deny(tmp_path: Path) -> None:
    root, _target = _windows_alias_workspace(tmp_path)
    runtime = Runtime.open("local", substrate=LocalResourceProviderSubstrate(root))
    try:
        denied_resource = runtime.filesystem.resource_for_path(_WINDOWS_STORED_PATH)
        assert (
            runtime.filesystem.resource_for_path(_WINDOWS_ALIAS_PATH)
            == denied_resource
        )
        pid = runtime.process.spawn(goal="enforce one Windows path authority")
        runtime.filesystem.grant_directory(
            pid,
            ".",
            [CapabilityRight.READ],
            issued_by="test",
        )
        runtime.capability.set_permission_policy(
            subject=pid,
            resource=denied_resource,
            rights=[CapabilityRight.READ],
            policy=CapabilityManager.ALWAYS_DENY,
            issued_by="test",
        )

        effects_before = tuple(runtime.store.list_external_effects(pid=pid))
        with pytest.raises(CapabilityDenied):
            runtime.filesystem.read_text(pid, _WINDOWS_ALIAS_PATH)
        assert tuple(runtime.store.list_external_effects(pid=pid)) == effects_before
    finally:
        runtime.close()


@pytest.mark.platform_windows
def test_windows_missing_case_aliases_share_create_and_directory_identity(
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("native Windows future-entry identity regression")
    root = tmp_path / "workspace"
    root.mkdir()
    provider = LocalFilesystemProvider(root)

    first = provider.resolve("Future/Report.TXT")
    second = provider.resolve("future/report.txt")
    assert first.relative == second.relative == "future/report.txt"

    provider.write_text(first, "created", "utf-8")
    created_alias = provider.resolve("FUTURE/REPORT.txt")
    assert created_alias.relative == first.relative
    assert provider.read_bytes(created_alias) == b"created"

    directory = provider.resolve("Another/Child")
    provider.make_directory(directory, parents=True, exist_ok=False)
    alias_directory = provider.resolve("ANOTHER/CHILD")
    assert alias_directory.relative == directory.relative == "another/child"
    assert provider.list_directory(provider.resolve("ANOTHER"))[0].name == "child"
    provider.delete_directory(alias_directory, recursive=True)
    assert provider.state(provider.resolve("another/child")).exists is False


@pytest.mark.platform_windows
def test_windows_native_case_sensitive_directories_preserve_identity_and_containment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "nt":
        pytest.skip("native Windows directory identity policy regression")
    case_parent = tmp_path / "case-sensitive-parent"
    case_parent.mkdir()
    try:
        _windows_enable_case_sensitive_directory(case_parent)
    except OSError as exc:
        pytest.skip(f"cannot enable native case-sensitive directory: {exc}")

    root = case_parent / "Workspace"
    sibling = case_parent / "workspace"
    root.mkdir()
    sibling.mkdir()
    try:
        _windows_enable_case_sensitive_directory(root)
    except OSError as exc:
        pytest.skip(f"cannot enable case sensitivity on provider root: {exc}")

    upper = root / "Report.txt"
    lower = root / "report.txt"
    upper.write_text("upper", encoding="utf-8")
    lower.write_text("lower", encoding="utf-8")
    outside = sibling / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    if os.path.samefile(upper, lower):
        pytest.fail("case-sensitive directory did not preserve distinct entries")

    provider = LocalFilesystemProvider(root)
    upper_resolved = provider.resolve("Report.txt")
    lower_resolved = provider.resolve("report.txt")
    assert upper_resolved.relative == "Report.txt"
    assert lower_resolved.relative == "report.txt"
    assert upper_resolved.relative != lower_resolved.relative
    assert provider.resolve("Future/Report.TXT").relative == "Future/Report.TXT"

    def forbid_outside_handle_open(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("outside sibling was inspected before containment denial")

    monkeypatch.setattr(
        local_substrate._WindowsDirectoryGuard,
        "open",
        forbid_outside_handle_open,
    )
    with pytest.raises(CapabilityDenied):
        provider.resolve(outside)


@pytest.mark.platform_windows
def test_windows_missing_path_denies_when_case_policy_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "nt":
        pytest.skip("native Windows directory identity policy regression")
    root = tmp_path / "workspace"
    root.mkdir()
    provider = LocalFilesystemProvider(root)

    def unavailable(_handle: int) -> bool:
        raise CapabilityDenied("simulated unavailable Windows case policy")

    monkeypatch.setattr(provider, "_windows_directory_case_sensitive", unavailable)
    with pytest.raises(CapabilityDenied, match="unavailable Windows case policy"):
        provider.resolve("missing.txt")

    assert list(root.iterdir()) == []


@pytest.mark.platform_windows
def test_windows_sink_rejects_case_rename_after_authorization(
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("native Windows sink identity regression")

    class CaseRenameProvider(LocalFilesystemProvider):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.renamed = False

        def _before_fallback_open(self, target: Path, flags: int) -> None:
            if self.renamed:
                return
            self.renamed = True
            intermediate = target.with_name("rename-intermediate.tmp")
            target.rename(intermediate)
            intermediate.rename(target.with_name("VICTIM.TXT"))

    root = tmp_path / "workspace"
    root.mkdir()
    (root / "Victim.txt").write_text("protected", encoding="utf-8")
    provider = CaseRenameProvider(root)
    authorized = provider.resolve("victim.TXT")
    assert authorized.relative == "Victim.txt"

    with pytest.raises(
        CapabilityDenied,
        match="changed while its parent was guarded",
    ):
        provider.read_bytes(authorized)

    assert provider.renamed is True
    assert [path.name for path in root.iterdir()] == ["Victim.txt"]
    assert (root / "Victim.txt").read_text(encoding="utf-8") == "protected"

    pre_entry_root = tmp_path / "pre-entry-case-workspace"
    pre_entry_root.mkdir()
    pre_entry_target = pre_entry_root / "Victim.txt"
    pre_entry_target.write_text("protected", encoding="utf-8")
    pre_entry_provider = LocalFilesystemProvider(pre_entry_root)
    pre_entry_authorized = pre_entry_provider.resolve("Victim.txt")
    intermediate = pre_entry_target.with_name("rename-intermediate.tmp")
    pre_entry_target.rename(intermediate)
    intermediate.rename(pre_entry_target.with_name("VICTIM.TXT"))

    with pytest.raises(CapabilityDenied, match="changed during validation"):
        pre_entry_provider.read_bytes(pre_entry_authorized)

    assert [path.name for path in pre_entry_root.iterdir()] == ["VICTIM.TXT"]
    assert (pre_entry_root / "VICTIM.TXT").read_text(encoding="utf-8") == "protected"


@pytest.mark.platform_windows
def test_windows_read_rejects_same_spelling_file_id_rebind(tmp_path: Path) -> None:
    if os.name != "nt":
        pytest.skip("native Windows FileId sink regression")

    class RebindBeforeOpenProvider(LocalFilesystemProvider):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.rebound = False

        def _before_fallback_open(self, target: Path, flags: int) -> None:
            if self.rebound:
                return
            self.rebound = True
            _windows_replace_file(
                target,
                target.with_name("original-victim.txt"),
                "replacement",
            )

    root = tmp_path / "workspace"
    root.mkdir()
    victim = root / "Victim.txt"
    victim.write_text("authorized", encoding="utf-8")
    provider = RebindBeforeOpenProvider(root)
    authorized = provider.resolve("Victim.txt")

    with pytest.raises(
        CapabilityDenied,
        match="changed while its parent was guarded",
    ):
        provider.read_bytes(authorized)

    assert provider.rebound is True
    assert victim.read_text(encoding="utf-8") == "authorized"
    assert not (root / "original-victim.txt").exists()

    pre_entry_root = tmp_path / "pre-entry-read-workspace"
    pre_entry_root.mkdir()
    pre_entry_victim = pre_entry_root / "Victim.txt"
    pre_entry_victim.write_text("authorized", encoding="utf-8")
    pre_entry_provider = LocalFilesystemProvider(pre_entry_root)
    pre_entry_authorized = pre_entry_provider.resolve("Victim.txt")
    _windows_replace_file(
        pre_entry_victim,
        pre_entry_root / "original-victim.txt",
        "replacement",
    )

    with pytest.raises(
        CapabilityDenied,
        match="object identity changed after authorization",
    ):
        pre_entry_provider.read_bytes(pre_entry_authorized)

    assert pre_entry_victim.read_text(encoding="utf-8") == "replacement"
    assert (pre_entry_root / "original-victim.txt").read_text(
        encoding="utf-8"
    ) == "authorized"


@pytest.mark.platform_windows
def test_windows_state_rejects_same_spelling_file_id_rebind(tmp_path: Path) -> None:
    if os.name != "nt":
        pytest.skip("native Windows FileId state regression")

    class RebindAtStateProvider(LocalFilesystemProvider):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.rebound = False

        def _before_path_sink(self, operation: str, target: Path) -> None:
            if operation != "state" or self.rebound:
                return
            self.rebound = True
            _windows_replace_file(
                target,
                target.with_name("original-state.txt"),
                "replacement-state",
            )

    root = tmp_path / "workspace"
    root.mkdir()
    victim = root / "state.txt"
    victim.write_text("authorized-state", encoding="utf-8")
    provider = RebindAtStateProvider(root)
    authorized = provider.resolve("state.txt")

    with pytest.raises(CapabilityDenied, match="changed before state"):
        provider.state(authorized)

    assert provider.rebound is True
    assert victim.read_text(encoding="utf-8") == "authorized-state"
    assert not (root / "original-state.txt").exists()

    pre_entry_root = tmp_path / "pre-entry-state-workspace"
    pre_entry_root.mkdir()
    pre_entry_victim = pre_entry_root / "state.txt"
    pre_entry_victim.write_text("authorized-state", encoding="utf-8")
    pre_entry_provider = LocalFilesystemProvider(pre_entry_root)
    pre_entry_authorized = pre_entry_provider.resolve("state.txt")
    _windows_replace_file(
        pre_entry_victim,
        pre_entry_root / "original-state.txt",
        "replacement-state",
    )

    with pytest.raises(
        CapabilityDenied,
        match="object identity changed after authorization",
    ):
        pre_entry_provider.state(pre_entry_authorized)

    assert pre_entry_victim.read_text(encoding="utf-8") == "replacement-state"
    assert (pre_entry_root / "original-state.txt").read_text(
        encoding="utf-8"
    ) == "authorized-state"


@pytest.mark.platform_windows
def test_windows_delete_file_rejects_same_spelling_file_id_rebind(
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("native Windows FileId delete regression")

    class RebindAtDeleteProvider(LocalFilesystemProvider):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.rebound = False

        def _before_path_sink(self, operation: str, target: Path) -> None:
            if operation != "delete_file" or self.rebound:
                return
            self.rebound = True
            _windows_replace_file(
                target,
                target.with_name("original-delete.txt"),
                "replacement-delete",
            )

    root = tmp_path / "workspace"
    root.mkdir()
    victim = root / "delete.txt"
    victim.write_text("authorized-delete", encoding="utf-8")
    provider = RebindAtDeleteProvider(root)
    authorized = provider.resolve("delete.txt")

    with pytest.raises(CapabilityDenied, match="changed before delete_file"):
        provider.delete_file(authorized)

    assert provider.rebound is True
    assert victim.read_text(encoding="utf-8") == "authorized-delete"
    assert not (root / "original-delete.txt").exists()

    pre_entry_root = tmp_path / "pre-entry-delete-workspace"
    pre_entry_root.mkdir()
    pre_entry_victim = pre_entry_root / "delete.txt"
    pre_entry_victim.write_text("authorized-delete", encoding="utf-8")
    pre_entry_provider = LocalFilesystemProvider(pre_entry_root)
    pre_entry_authorized = pre_entry_provider.resolve("delete.txt")
    _windows_replace_file(
        pre_entry_victim,
        pre_entry_root / "original-delete.txt",
        "replacement-delete",
    )

    with pytest.raises(
        CapabilityDenied,
        match="object identity changed after authorization",
    ):
        pre_entry_provider.delete_file(pre_entry_authorized)

    assert pre_entry_victim.read_text(encoding="utf-8") == "replacement-delete"
    assert (pre_entry_root / "original-delete.txt").read_text(
        encoding="utf-8"
    ) == "authorized-delete"


@pytest.mark.platform_windows
def test_windows_recursive_delete_rejects_same_spelling_directory_id_rebind(
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("native Windows FileId recursive-delete regression")

    class RebindAtRecursiveDeleteProvider(LocalFilesystemProvider):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.rebound = False

        def _before_path_sink(self, operation: str, target: Path) -> None:
            if operation != "delete_directory" or self.rebound:
                return
            self.rebound = True
            target.rename(target.with_name("original-tree"))
            target.mkdir()
            (target / "replacement-marker.txt").write_text(
                "replacement",
                encoding="utf-8",
            )

    root = tmp_path / "workspace"
    root.mkdir()
    tree = root / "tree"
    tree.mkdir()
    (tree / "authorized-marker.txt").write_text("authorized", encoding="utf-8")
    provider = RebindAtRecursiveDeleteProvider(root)
    authorized = provider.resolve("tree")

    with pytest.raises(
        CapabilityDenied,
        match="changed before delete_directory",
    ):
        provider.delete_directory(authorized, recursive=True)

    assert provider.rebound is True
    assert (tree / "authorized-marker.txt").read_text(
        encoding="utf-8"
    ) == "authorized"
    assert not (tree / "replacement-marker.txt").exists()
    assert not (root / "original-tree").exists()

    pre_entry_root = tmp_path / "pre-entry-tree-workspace"
    pre_entry_tree = pre_entry_root / "tree"
    pre_entry_tree.mkdir(parents=True)
    (pre_entry_tree / "authorized-marker.txt").write_text(
        "authorized",
        encoding="utf-8",
    )
    pre_entry_provider = LocalFilesystemProvider(pre_entry_root)
    pre_entry_authorized = pre_entry_provider.resolve("tree")
    pre_entry_original = pre_entry_root / "original-tree"
    pre_entry_tree.rename(pre_entry_original)
    pre_entry_tree.mkdir()
    (pre_entry_tree / "replacement-marker.txt").write_text(
        "replacement",
        encoding="utf-8",
    )

    with pytest.raises(
        CapabilityDenied,
        match="object identity changed after authorization",
    ):
        pre_entry_provider.delete_directory(pre_entry_authorized, recursive=True)

    assert (pre_entry_tree / "replacement-marker.txt").read_text(
        encoding="utf-8"
    ) == "replacement"
    assert (pre_entry_original / "authorized-marker.txt").read_text(
        encoding="utf-8"
    ) == "authorized"


@pytest.mark.platform_windows
def test_windows_recursive_delete_holds_nested_directory_handle_until_delete(
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("native Windows nested recursive-delete timing regression")

    class NestedRebindAttemptProvider(LocalFilesystemProvider):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.rebind_blocked = False

        def _after_protected_delete_entry(self, target: Path) -> None:
            nested = self.root / "tree" / "nested"
            if self.rebind_blocked or target.parent != nested:
                return
            try:
                nested.rename(nested.with_name("original-nested"))
            except OSError:
                self.rebind_blocked = True
                return
            nested.mkdir()
            (nested / "replacement.txt").write_text(
                "replacement",
                encoding="utf-8",
            )

    root = tmp_path / "workspace"
    nested = root / "tree" / "nested"
    nested.mkdir(parents=True)
    (nested / "authorized.txt").write_text("authorized", encoding="utf-8")
    provider = NestedRebindAttemptProvider(root)

    provider.delete_directory(provider.resolve("tree"), recursive=True)

    assert provider.rebind_blocked is True
    assert not (root / "tree").exists()
    assert not (root / "tree" / "original-nested").exists()


@pytest.mark.platform_windows
def test_windows_make_directory_rejects_existing_parent_file_id_rebind(
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("native Windows parent FileId create regression")

    class RebindParentProvider(LocalFilesystemProvider):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.rebound = False

        def _before_path_sink(self, operation: str, target: Path) -> None:
            if operation != "make_directory" or self.rebound:
                return
            self.rebound = True
            target.parent.rename(target.parent.with_name("original-parent"))
            target.parent.mkdir()

    root = tmp_path / "workspace"
    root.mkdir()
    parent = root / "parent"
    parent.mkdir()
    provider = RebindParentProvider(root)
    authorized = provider.resolve("parent/child")

    with pytest.raises(CapabilityDenied, match="changed before make_directory"):
        provider.make_directory(authorized, parents=True, exist_ok=False)

    assert provider.rebound is True
    assert parent.is_dir()
    assert not (parent / "child").exists()
    assert not (root / "original-parent").exists()

    pre_entry_root = tmp_path / "pre-entry-mkdir-workspace"
    pre_entry_parent = pre_entry_root / "parent"
    pre_entry_parent.mkdir(parents=True)
    pre_entry_provider = LocalFilesystemProvider(pre_entry_root)
    pre_entry_authorized = pre_entry_provider.resolve("parent/child")
    pre_entry_original = pre_entry_root / "original-parent"
    pre_entry_parent.rename(pre_entry_original)
    pre_entry_parent.mkdir()

    with pytest.raises(
        CapabilityDenied,
        match="parent identity changed after authorization",
    ):
        pre_entry_provider.make_directory(
            pre_entry_authorized,
            parents=True,
            exist_ok=False,
        )

    assert not (pre_entry_parent / "child").exists()
    assert pre_entry_original.is_dir()


@pytest.mark.platform_windows
def test_windows_write_rejects_existing_parent_file_id_rebind(
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("native Windows parent FileId write regression")

    class RebindWriteParentProvider(LocalFilesystemProvider):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.rebound = False

        def _before_path_sink(self, operation: str, target: Path) -> None:
            if operation != "write_parent" or self.rebound:
                return
            self.rebound = True
            target.rename(target.with_name("original-write-parent"))
            target.mkdir()

    root = tmp_path / "workspace"
    root.mkdir()
    parent = root / "parent"
    parent.mkdir()
    provider = RebindWriteParentProvider(root)
    authorized = provider.resolve("parent/child.txt")

    with pytest.raises(CapabilityDenied, match="changed before write_parent"):
        provider.write_text(authorized, "blocked", "utf-8")

    assert provider.rebound is True
    assert parent.is_dir()
    assert not (parent / "child.txt").exists()
    assert not (root / "original-write-parent").exists()

    pre_entry_root = tmp_path / "pre-entry-write-workspace"
    pre_entry_parent = pre_entry_root / "parent"
    pre_entry_parent.mkdir(parents=True)
    pre_entry_provider = LocalFilesystemProvider(pre_entry_root)
    pre_entry_authorized = pre_entry_provider.resolve("parent/child.txt")
    pre_entry_original = pre_entry_root / "original-write-parent"
    pre_entry_parent.rename(pre_entry_original)
    pre_entry_parent.mkdir()

    with pytest.raises(
        CapabilityDenied,
        match="parent identity changed after authorization",
    ):
        pre_entry_provider.write_text(
            pre_entry_authorized,
            "blocked",
            "utf-8",
        )

    assert not (pre_entry_parent / "child.txt").exists()
    assert pre_entry_original.is_dir()


@pytest.mark.platform_windows
def test_windows_verified_root_guard_blocks_rebind_before_all_path_sinks(
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("native Windows root FileId sink regression")

    class RootRebindProvider(LocalFilesystemProvider):
        def __init__(self, root: Path, operation: str) -> None:
            super().__init__(root)
            self.operation = operation
            self.attempted = False
            self.blocked = False

        def _before_path_sink(self, operation: str, target: Path) -> None:
            if operation != self.operation or self.attempted:
                return
            self.attempted = True
            backup = self.root.with_name(f"{self.root.name}-original")
            anchor_name = "anchor"
            try:
                self.root.rename(backup)
                self.root.mkdir()
                (backup / anchor_name).rename(self.root / anchor_name)
            except OSError as exc:
                self.blocked = True
                raise CapabilityDenied(
                    "verified root guard blocked the test rebind"
                ) from exc
            raise AssertionError("verified root guard allowed a root rebind")

    operations = (
        ("state", "state", "anchor/victim.txt"),
        ("read", "read_bytes", "anchor/victim.txt"),
        ("write", "write_parent", "anchor/new-file.txt"),
        ("compare-and-swap", "write_parent", "anchor/victim.txt"),
        ("make-directory", "make_directory", "anchor/new-directory"),
        ("list-directory", "list_directory", "anchor"),
        ("delete-file", "delete_file", "anchor/victim.txt"),
        ("contains-descendant", None, "anchor"),
        ("delete-directory", "delete_directory", "anchor/tree"),
        (
            "delete-directory-protected",
            "delete_directory",
            "anchor/protected-tree",
        ),
    )

    def invoke(
        provider: LocalFilesystemProvider,
        case_name: str,
        authorized: object,
    ) -> None:
        if case_name == "state":
            provider.state(authorized)  # type: ignore[arg-type]
        elif case_name == "read":
            provider.read_bytes(authorized)  # type: ignore[arg-type]
        elif case_name == "write":
            provider.write_text(  # type: ignore[arg-type]
                authorized,
                "blocked",
                "utf-8",
            )
        elif case_name == "compare-and-swap":
            provider.write_text_compare_and_swap(  # type: ignore[arg-type]
                authorized,
                "blocked",
                "utf-8",
                expected_content_sha256=hashlib.sha256(b"protected").hexdigest(),
            )
        elif case_name == "make-directory":
            provider.make_directory(  # type: ignore[arg-type]
                authorized,
                parents=True,
                exist_ok=False,
            )
        elif case_name == "list-directory":
            provider.list_directory(authorized)  # type: ignore[arg-type]
        elif case_name == "delete-file":
            provider.delete_file(authorized)  # type: ignore[arg-type]
        elif case_name == "contains-descendant":
            provider.contains_descendant_name(  # type: ignore[arg-type]
                authorized,
                names=("victim.txt",),
            )
        elif case_name == "delete-directory":
            provider.delete_directory(  # type: ignore[arg-type]
                authorized,
                recursive=True,
            )
        elif case_name == "delete-directory-protected":
            provider.delete_directory_protected(  # type: ignore[arg-type]
                authorized,
                recursive=True,
                protected_descendant_names=("metadata",),
            )
        else:  # pragma: no cover - operation table is closed above
            raise AssertionError(f"unknown root-rebind case: {case_name}")

    def prepare_root(root: Path) -> Path:
        anchor = root / "anchor"
        anchor.mkdir(parents=True)
        (anchor / "victim.txt").write_text("protected", encoding="utf-8")
        tree = anchor / "tree"
        tree.mkdir()
        (tree / "child.txt").write_text("protected", encoding="utf-8")
        protected_tree = anchor / "protected-tree"
        protected_tree.mkdir()
        (protected_tree / "metadata").write_text("protected", encoding="utf-8")
        return anchor

    def assert_preserved(anchor: Path) -> None:
        assert (anchor / "victim.txt").read_text(encoding="utf-8") == "protected"
        assert (anchor / "tree" / "child.txt").read_text(
            encoding="utf-8"
        ) == "protected"
        assert (anchor / "protected-tree" / "metadata").read_text(
            encoding="utf-8"
        ) == "protected"
        assert not (anchor / "new-directory").exists()
        assert not (anchor / "new-file.txt").exists()

    for case_name, hook_operation, relative in operations:
        if hook_operation is not None:
            case = tmp_path / f"{case_name}-hook"
            root = case / "workspace"
            anchor = prepare_root(root)
            provider = RootRebindProvider(root, hook_operation)
            authorized = provider.resolve(relative)

            with pytest.raises(CapabilityDenied):
                invoke(provider, case_name, authorized)

            assert provider.attempted is True
            assert provider.blocked is True
            assert not root.with_name("workspace-original").exists()
            assert_preserved(anchor)

        race_case = tmp_path / f"{case_name}-pre-entry"
        race_root = race_case / "workspace"
        prepare_root(race_root)
        race_provider = LocalFilesystemProvider(race_root)
        race_authorized = race_provider.resolve(relative)
        original_root = race_root.with_name("workspace-original")
        race_root.rename(original_root)
        race_root.mkdir()
        (original_root / "anchor").rename(race_root / "anchor")

        with pytest.raises(CapabilityDenied):
            invoke(race_provider, case_name, race_authorized)

        moved_anchor = race_root / "anchor"
        assert_preserved(moved_anchor)


@pytest.mark.platform_windows
def test_windows_read_only_guards_block_reparse_mutation_across_path_sinks(
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("native Windows guarded reparse mutation regression")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("outside", encoding="utf-8")
    probe = tmp_path / "junction-probe"
    probe.mkdir()
    try:
        _windows_mutate_directory_to_junction(probe, outside)
    except OSError as exc:
        pytest.skip(f"cannot exercise native reparse mutation: {exc}")
    assert (probe / "secret.txt").read_text(encoding="utf-8") == "outside"
    os.rmdir(probe)

    class ReparseMutationAttemptProvider(LocalFilesystemProvider):
        def __init__(
            self,
            root: Path,
            operation: str,
            outside: Path,
        ) -> None:
            super().__init__(root)
            self.operation = operation
            self.outside = outside
            self.blocked = False

        def _after_windows_path_guarded(
            self,
            operation: str,
            target: Path,
        ) -> None:
            if operation != self.operation or self.blocked:
                return
            try:
                _windows_mutate_directory_to_junction(target, self.outside)
            except OSError as exc:
                if getattr(exc, "winerror", None) != 32:
                    raise
                self.blocked = True
                return
            raise AssertionError(
                "guard allowed same-FileId directory reparse mutation"
            )

    writer_root = tmp_path / "preexisting-writer"
    writer_root.mkdir()
    writer_provider = LocalFilesystemProvider(writer_root)
    kernel32, writer_handle = _windows_open_directory_reparse_writer(writer_root)
    close_writer = kernel32.CloseHandle
    close_writer.argtypes = [ctypes.c_void_p]
    close_writer.restype = ctypes.c_int
    try:
        with pytest.raises(CapabilityDenied):
            writer_provider.resolve("missing.txt")
    finally:
        close_writer(writer_handle)
    assert writer_provider.resolve("missing.txt").relative == "missing.txt"

    create_root = tmp_path / "create-root"
    create_root.mkdir()
    create_provider = ReparseMutationAttemptProvider(
        create_root,
        "open_parent",
        outside,
    )
    create_provider.write_text(
        create_provider.resolve("payload.txt"),
        "inside",
        "utf-8",
    )
    assert create_provider.blocked is True
    assert (create_root / "payload.txt").read_text(encoding="utf-8") == "inside"
    assert not (outside / "payload.txt").exists()

    mkdir_root = tmp_path / "mkdir-root"
    mkdir_root.mkdir()
    mkdir_provider = ReparseMutationAttemptProvider(
        mkdir_root,
        "make_directory_parent",
        outside,
    )
    mkdir_provider.make_directory(
        mkdir_provider.resolve("child"),
        parents=True,
        exist_ok=False,
    )
    assert mkdir_provider.blocked is True
    assert (mkdir_root / "child").is_dir()
    assert not (outside / "child").exists()

    list_root = tmp_path / "list-root"
    listed = list_root / "listed"
    listed.mkdir(parents=True)
    list_provider = ReparseMutationAttemptProvider(
        list_root,
        "list_directory",
        outside,
    )
    assert list_provider.list_directory(list_provider.resolve("listed")) == []
    assert list_provider.blocked is True

    delete_root = tmp_path / "delete-root"
    deleted = delete_root / "deleted"
    deleted.mkdir(parents=True)
    delete_provider = ReparseMutationAttemptProvider(
        delete_root,
        "delete_directory",
        outside,
    )
    delete_provider.delete_directory(
        delete_provider.resolve("deleted"),
        recursive=True,
    )
    assert delete_provider.blocked is True
    assert not deleted.exists()
    assert (outside / "secret.txt").read_text(encoding="utf-8") == "outside"


@pytest.mark.platform_windows
def test_windows_descendant_junctions_never_escape_guarded_recursive_scan(
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("native Windows descendant junction scan regression")

    outside = tmp_path / "outside"
    outside.mkdir()
    protected_name = "outside-only-secret.txt"
    (outside / protected_name).write_text("outside", encoding="utf-8")

    probe = tmp_path / "junction-probe"
    probe.mkdir()
    try:
        _windows_mutate_directory_to_junction(probe, outside)
    except OSError as exc:
        pytest.skip(f"cannot exercise native descendant junction scan: {exc}")
    assert (probe / protected_name).read_text(encoding="utf-8") == "outside"
    os.rmdir(probe)

    direct_root = tmp_path / "direct-workspace"
    direct_tree = direct_root / "tree"
    direct_junction = direct_tree / "outside-link"
    direct_junction.mkdir(parents=True)
    _windows_mutate_directory_to_junction(direct_junction, outside)
    direct_provider = LocalFilesystemProvider(direct_root)

    assert direct_provider.contains_descendant_name(
        direct_provider.resolve("tree"),
        names=(protected_name,),
    ) is False
    assert (outside / protected_name).read_text(encoding="utf-8") == "outside"
    os.rmdir(direct_junction)

    class DescendantRaceProvider(LocalFilesystemProvider):
        def __init__(self, root: Path, phase: str) -> None:
            super().__init__(root)
            self.phase = phase
            self.swapped = False
            self.mutation_blocked = False

        def _after_windows_descendant_enumerated(self, target: Path) -> None:
            if self.phase != "before-guard" or target.name != "candidate":
                return
            _windows_mutate_directory_to_junction(target, outside)
            self.swapped = True

        def _after_windows_descendant_guarded(self, target: Path) -> None:
            if self.phase != "while-guarded" or target.name != "candidate":
                return
            try:
                _windows_mutate_directory_to_junction(target, outside)
            except OSError as exc:
                if getattr(exc, "winerror", None) != 32:
                    raise
                self.mutation_blocked = True
                return
            raise AssertionError(
                "descendant guard allowed same-FileId reparse mutation"
            )

    race_root = tmp_path / "race-workspace"
    race_candidate = race_root / "tree" / "candidate"
    race_candidate.mkdir(parents=True)
    race_provider = DescendantRaceProvider(race_root, "before-guard")

    assert race_provider.contains_descendant_name(
        race_provider.resolve("tree"),
        names=(protected_name,),
    ) is False
    assert race_provider.swapped is True
    assert (outside / protected_name).read_text(encoding="utf-8") == "outside"
    os.rmdir(race_candidate)

    guarded_root = tmp_path / "guarded-workspace"
    guarded_candidate = guarded_root / "tree" / "candidate"
    guarded_candidate.mkdir(parents=True)
    guarded_provider = DescendantRaceProvider(guarded_root, "while-guarded")

    assert guarded_provider.contains_descendant_name(
        guarded_provider.resolve("tree"),
        names=(protected_name,),
    ) is False
    assert guarded_provider.mutation_blocked is True
    assert guarded_candidate.is_dir()
    assert (outside / protected_name).read_text(encoding="utf-8") == "outside"


@pytest.mark.platform_windows
def test_windows_parent_creation_rejects_future_component_appearance(
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("native Windows future parent appearance regression")

    class MaterializeFutureParentProvider(LocalFilesystemProvider):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.materialized = False

        def _before_path_sink(self, operation: str, target: Path) -> None:
            if operation != "make_directory" or self.materialized:
                return
            self.materialized = True
            target.parent.mkdir()

    root = tmp_path / "workspace"
    root.mkdir()
    provider = MaterializeFutureParentProvider(root)
    authorized = provider.resolve("future/child")

    with pytest.raises(CapabilityDenied):
        provider.make_directory(authorized, parents=True, exist_ok=False)

    assert (root / "future").is_dir()
    assert not (root / "future" / "child").exists()


@pytest.mark.platform_windows
def test_windows_missing_directory_appearance_is_rejected_with_exist_ok(
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("native Windows missing directory appearance regression")

    class MaterializeTargetProvider(LocalFilesystemProvider):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.materialized = False

        def _before_path_sink(self, operation: str, target: Path) -> None:
            if operation != "make_directory" or self.materialized:
                return
            self.materialized = True
            target.mkdir()

    root = tmp_path / "workspace"
    root.mkdir()
    provider = MaterializeTargetProvider(root)
    authorized = provider.resolve("appeared")

    with pytest.raises(CapabilityDenied):
        provider.make_directory(authorized, parents=True, exist_ok=True)

    assert (root / "appeared").is_dir()


@pytest.mark.platform_windows
def test_windows_handle_delete_controls_file_empty_and_recursive_directory(
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("native Windows handle-delete control regression")
    assert ctypes.sizeof(local_substrate._FILE_DISPOSITION_INFO) == 1

    root = tmp_path / "workspace"
    root.mkdir()
    file_target = root / "file.txt"
    file_target.write_text("delete", encoding="utf-8")
    empty = root / "empty"
    empty.mkdir()
    tree = root / "tree"
    (tree / "nested").mkdir(parents=True)
    (tree / "nested" / "file.txt").write_text("delete", encoding="utf-8")
    provider = LocalFilesystemProvider(root)

    provider.delete_file(provider.resolve("file.txt"))
    provider.delete_directory(provider.resolve("empty"), recursive=False)
    provider.delete_directory(provider.resolve("tree"), recursive=True)

    assert not file_target.exists()
    assert not empty.exists()
    assert not tree.exists()


@pytest.mark.platform_windows
def test_windows_provider_and_substrate_reject_reparse_root_before_following_it(
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("native Windows root reparse regression")
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    junction = tmp_path / "junction-root"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", os.fspath(junction), os.fspath(real_root)],
        check=False,
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        pytest.skip("cannot create a native Windows junction for root validation")

    with pytest.raises(CapabilityDenied, match="root.*reparse point"):
        LocalFilesystemProvider(junction)
    with pytest.raises(CapabilityDenied, match="root.*reparse point"):
        LocalResourceProviderSubstrate(junction)


def _windows_short_path(path: Path) -> Path:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_short_path_name = kernel32.GetShortPathNameW
    get_short_path_name.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
    ]
    get_short_path_name.restype = ctypes.c_uint32
    size = get_short_path_name(os.fspath(path), None, 0)
    if size == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_unicode_buffer(size)
    result = get_short_path_name(os.fspath(path), buffer, size)
    if result == 0 or result >= size:
        raise ctypes.WinError(ctypes.get_last_error())
    return Path(buffer.value)


@pytest.mark.platform_windows
def test_windows_dos_83_alias_expands_to_stored_long_identity_when_available(
    tmp_path: Path,
) -> None:
    if os.name != "nt":
        pytest.skip("native Windows DOS 8.3 identity regression")
    root = tmp_path / "WorkspaceLongIdentityName"
    root.mkdir()
    target = root / "EvidenceLongIdentityName.txt"
    target.write_text("evidence", encoding="utf-8")
    provider = LocalFilesystemProvider(root)

    short_root = _windows_short_path(root)
    short_target = _windows_short_path(target)
    alias = short_target.relative_to(short_root).as_posix()
    if (
        os.path.normcase(alias) == os.path.normcase(target.name)
        or "~" not in alias
    ):
        pytest.skip("test volume did not produce a DOS 8.3 alias")

    assert provider.resolve(alias).relative == target.name
    assert provider.read_bytes(provider.resolve(alias)) == b"evidence"


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
