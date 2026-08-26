from __future__ import annotations

import asyncio
import os
import stat
import tempfile
from pathlib import Path

import pytest

from agent_libos import Runtime
from agent_libos.config import AgentLibOSConfig, RuntimeDefaults
from agent_libos.models.exceptions import ValidationError
from agent_libos.runtime import RuntimeBuilder
from agent_libos.runtime.task_runs import TaskRunManager
from agent_libos.runtime.windows_store_identity import (
    LegacyWindowsStoreIdentityError,
    WindowsStoreIdentityValidationSummary,
    validate_legacy_windows_store_identities,
)
from agent_libos.storage import (
    SQLiteStore,
    display_store_target,
    open_store,
    resolve_store_target,
)
from agent_libos.substrate import LocalResourceProviderSubstrate


def _set_user_home(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))


def _sqlite_artifacts(directory: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in directory.rglob("*")
        if path.is_file() or path.is_symlink()
    )


def _sqlite_identity_lease_inventory() -> tuple[bool, tuple[str, ...]]:
    uid = os.getuid() if hasattr(os, "getuid") else 0
    directory = (
        Path(tempfile.gettempdir()).resolve()
        / f"agent-libos-sqlite-leases-{uid}"
    )
    if not directory.exists():
        return False, ()
    return True, tuple(sorted(path.name for path in directory.iterdir()))


@pytest.mark.parametrize(
    "target_factory",
    (
        lambda workspace: Path("runtime.sqlite"),
        lambda workspace: workspace / "nested" / "runtime.sqlite",
        lambda workspace: f"sqlite:////{(workspace / 'uri.sqlite').as_posix().lstrip('/')}",
    ),
    ids=("relative", "absolute", "sqlite-uri"),
)
def test_runtime_store_overlap_rejected_before_sqlite_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target_factory: object,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    target = target_factory(workspace)  # type: ignore[operator]
    leases_before = _sqlite_identity_lease_inventory()

    with pytest.raises(
        ValidationError,
        match="persistent SQLite runtime store must be outside",
    ):
        Runtime.open(target)

    assert _sqlite_artifacts(workspace) == ()
    assert _sqlite_identity_lease_inventory() == leases_before


def test_configured_runtime_store_overlap_rejected_before_sqlite_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    config = AgentLibOSConfig(
        runtime=RuntimeDefaults(local_store_target="state/runtime.sqlite")
    )
    leases_before = _sqlite_identity_lease_inventory()

    with pytest.raises(
        ValidationError,
        match="persistent SQLite runtime store must be outside",
    ):
        Runtime.open(config=config)

    assert _sqlite_artifacts(workspace) == ()
    assert _sqlite_identity_lease_inventory() == leases_before


def test_default_substrate_is_not_allocated_for_store_overlap_denial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    allocations: list[Path] = []

    class TrackingSubstrate(LocalResourceProviderSubstrate):
        def __init__(self, workspace_root: str | Path, *args: object, **kwargs: object):
            allocations.append(Path(workspace_root))
            super().__init__(workspace_root, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        "agent_libos.runtime.builder.LocalResourceProviderSubstrate",
        TrackingSubstrate,
    )

    with pytest.raises(
        ValidationError,
        match="persistent SQLite runtime store must be outside",
    ):
        Runtime.open(workspace / "runtime.sqlite")

    assert allocations == []
    assert _sqlite_artifacts(workspace) == ()


def test_default_substrate_is_not_allocated_in_invalid_sync_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    allocations: list[Path] = []

    class TrackingSubstrate(LocalResourceProviderSubstrate):
        def __init__(self, workspace_root: str | Path, *args: object, **kwargs: object):
            allocations.append(Path(workspace_root))
            super().__init__(workspace_root, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        "agent_libos.runtime.builder.LocalResourceProviderSubstrate",
        TrackingSubstrate,
    )

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="active event loop"):
            Runtime.open("local")

    asyncio.run(exercise())

    assert allocations == []


def test_explicit_workspace_store_overlap_rejected_before_sqlite_side_effects(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    substrate = LocalResourceProviderSubstrate(workspace)
    target = workspace / "state" / "runtime.sqlite"
    leases_before = _sqlite_identity_lease_inventory()

    with pytest.raises(
        ValidationError,
        match="persistent SQLite runtime store must be outside",
    ):
        Runtime.open(target, substrate=substrate)

    assert _sqlite_artifacts(workspace) == ()
    assert _sqlite_identity_lease_inventory() == leases_before


def test_preexisting_workspace_store_denial_is_byte_and_mode_preserving(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "runtime.sqlite"
    seeded = SQLiteStore(target)
    seeded.close()
    before_bytes = target.read_bytes()
    before_mode = stat.S_IMODE(os.lstat(target).st_mode)
    before_artifacts = _sqlite_artifacts(workspace)
    leases_before = _sqlite_identity_lease_inventory()

    with pytest.raises(
        ValidationError,
        match="persistent SQLite runtime store must be outside",
    ):
        Runtime.open(
            target,
            substrate=LocalResourceProviderSubstrate(workspace),
        )

    assert target.read_bytes() == before_bytes
    assert stat.S_IMODE(os.lstat(target).st_mode) == before_mode
    assert _sqlite_artifacts(workspace) == before_artifacts
    assert _sqlite_identity_lease_inventory() == leases_before


@pytest.mark.skipif(
    not hasattr(os, "symlink"),
    reason="host does not expose symlink creation",
)
def test_runtime_store_overlap_alias_rejected_before_sqlite_side_effects(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    alias = outside / "workspace-alias"
    try:
        alias.symlink_to(workspace, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"host cannot create directory symlinks: {exc}")
    substrate = LocalResourceProviderSubstrate(workspace)
    leases_before = _sqlite_identity_lease_inventory()

    with pytest.raises(
        ValidationError,
        match="persistent SQLite runtime",
    ):
        Runtime.open(alias / "runtime.sqlite", substrate=substrate)

    assert _sqlite_artifacts(workspace) == ()
    assert _sqlite_identity_lease_inventory() == leases_before


def test_runtime_store_workspace_case_alias_is_rejected_without_side_effects(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "CaseSensitiveProbe"
    workspace.mkdir()
    case_alias = workspace.with_name(workspace.name.swapcase())
    try:
        workspace_stat = os.stat(workspace)
        alias_stat = os.stat(case_alias)
    except FileNotFoundError:
        pytest.skip("temporary filesystem is case-sensitive")
    if (workspace_stat.st_dev, workspace_stat.st_ino) != (
        alias_stat.st_dev,
        alias_stat.st_ino,
    ):
        pytest.skip("case-variant spelling does not alias the workspace")
    target = case_alias / "runtime.sqlite"
    leases_before = _sqlite_identity_lease_inventory()

    with pytest.raises(
        ValidationError,
        match="persistent SQLite runtime store must be outside",
    ):
        Runtime.open(
            target,
            substrate=LocalResourceProviderSubstrate(workspace),
        )

    assert _sqlite_artifacts(workspace) == ()
    assert _sqlite_identity_lease_inventory() == leases_before


def test_dangling_sqlite_symlink_is_rejected_without_target_or_sidecar_creation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "missing" / "runtime.sqlite"
    alias = workspace / "runtime-alias.sqlite"
    try:
        alias.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"host cannot create file symlinks: {exc}")
    substrate = LocalResourceProviderSubstrate(workspace)
    leases_before = _sqlite_identity_lease_inventory()

    with pytest.raises(
        ValidationError,
        match="persistent SQLite runtime store must be outside",
    ):
        Runtime.open(alias, substrate=substrate)

    assert alias.is_symlink()
    assert not target.exists()
    assert not target.parent.exists()
    assert tuple(workspace.iterdir()) == (alias,)
    assert _sqlite_identity_lease_inventory() == leases_before


def test_validated_sqlite_alias_is_frozen_before_store_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    external = tmp_path / "external"
    aliases = tmp_path / "aliases"
    workspace.mkdir()
    external.mkdir()
    aliases.mkdir()
    original_target = external / "runtime.sqlite"
    retargeted = workspace / "retargeted.sqlite"
    alias = aliases / "runtime-alias.sqlite"
    try:
        alias.symlink_to(original_target)
    except OSError as exc:
        pytest.skip(f"host cannot create file symlinks: {exc}")
    substrate = LocalResourceProviderSubstrate(workspace)
    real_open_store = open_store

    def retarget_then_open(target: object, **kwargs: object) -> object:
        alias.unlink()
        alias.symlink_to(retargeted)
        return real_open_store(target, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        "agent_libos.runtime.builder.open_store",
        retarget_then_open,
    )
    runtime = Runtime.open(alias, substrate=substrate)
    try:
        assert Path(runtime.store.path) == original_target
        assert original_target.exists()
        assert not retargeted.exists()
    finally:
        runtime.close()


def test_frozen_sqlite_missing_parent_rebind_is_rejected_before_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    external = tmp_path / "external"
    workspace.mkdir()
    external.mkdir()
    future_parent = external / "future"
    target = future_parent / "runtime.sqlite"
    substrate = LocalResourceProviderSubstrate(workspace)
    real_open_store = open_store
    leases_before = _sqlite_identity_lease_inventory()

    def rebind_parent_then_open(target: object, **kwargs: object) -> object:
        try:
            future_parent.symlink_to(workspace, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"host cannot create directory symlinks: {exc}")
        return real_open_store(target, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        "agent_libos.runtime.builder.open_store",
        rebind_parent_then_open,
    )

    with pytest.raises(
        ValidationError,
        match="unsafe frozen SQLite database path changed",
    ):
        Runtime.open(target, substrate=substrate)

    assert future_parent.is_symlink()
    assert _sqlite_artifacts(workspace) == ()
    assert _sqlite_identity_lease_inventory() == leases_before


def test_workspace_sqlite_alias_to_external_store_is_rejected(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    external = tmp_path / "external"
    workspace.mkdir()
    external.mkdir()
    target = external / "runtime.sqlite"
    alias = workspace / "runtime-alias.sqlite"
    try:
        alias.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"host cannot create file symlinks: {exc}")
    leases_before = _sqlite_identity_lease_inventory()

    with pytest.raises(
        ValidationError,
        match="persistent SQLite runtime store must be outside",
    ):
        Runtime.open(
            alias,
            substrate=LocalResourceProviderSubstrate(workspace),
        )

    assert alias.is_symlink()
    assert not target.exists()
    assert tuple(workspace.iterdir()) == (alias,)
    assert _sqlite_identity_lease_inventory() == leases_before


@pytest.mark.parametrize("async_open", (False, True), ids=("sync", "async"))
def test_default_workspace_is_frozen_before_store_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    async_open: bool,
) -> None:
    initial_workspace = tmp_path / "initial-workspace"
    state_directory = tmp_path / "state"
    initial_workspace.mkdir()
    state_directory.mkdir()
    target = state_directory / "runtime.sqlite"
    monkeypatch.chdir(initial_workspace)
    real_open_store = open_store

    def change_cwd_then_open(target: object, **kwargs: object) -> object:
        monkeypatch.chdir(state_directory)
        return real_open_store(target, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        "agent_libos.runtime.builder.open_store",
        change_cwd_then_open,
    )

    if async_open:
        async def exercise() -> None:
            runtime = await Runtime.aopen(target)
            try:
                assert Path(runtime.workspace_root) == initial_workspace.resolve()
                assert Path(runtime.store.path) == target
            finally:
                await runtime.ashutdown()

        asyncio.run(exercise())
    else:
        runtime = Runtime.open(target)
        try:
            assert Path(runtime.workspace_root) == initial_workspace.resolve()
            assert Path(runtime.store.path) == target
        finally:
            runtime.close()


def test_async_relative_store_target_uses_public_entry_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    initial_directory = tmp_path / "initial"
    changed_directory = tmp_path / "changed"
    workspace = tmp_path / "workspace"
    initial_directory.mkdir()
    changed_directory.mkdir()
    workspace.mkdir()
    monkeypatch.chdir(initial_directory)
    original_capture = RuntimeBuilder._capture_owned_runtime_assembly

    def change_cwd_before_worker_resolution(
        builder: RuntimeBuilder[Runtime],
        target: str | Path | None,
        handshake: object,
        caller_loop: object,
    ) -> object:
        monkeypatch.chdir(changed_directory)
        return original_capture(
            builder,
            target,
            handshake,  # type: ignore[arg-type]
            caller_loop,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(
        RuntimeBuilder,
        "_capture_owned_runtime_assembly",
        change_cwd_before_worker_resolution,
    )

    async def exercise() -> None:
        runtime = await Runtime.aopen(
            "runtime.sqlite",
            substrate=LocalResourceProviderSubstrate(workspace),
        )
        try:
            assert Path(runtime.store.path) == initial_directory / "runtime.sqlite"
        finally:
            await runtime.ashutdown()

    asyncio.run(exercise())

    assert (initial_directory / "runtime.sqlite").exists()
    assert not (changed_directory / "runtime.sqlite").exists()


def test_async_runtime_store_overlap_rejected_before_sqlite_side_effects(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    substrate = LocalResourceProviderSubstrate(workspace)
    target = workspace / "async.sqlite"
    leases_before = _sqlite_identity_lease_inventory()

    async def exercise() -> None:
        with pytest.raises(
            ValidationError,
            match="persistent SQLite runtime store must be outside",
        ):
            await Runtime.aopen(target, substrate=substrate)

    asyncio.run(exercise())

    assert _sqlite_artifacts(workspace) == ()
    assert _sqlite_identity_lease_inventory() == leases_before


def test_caller_owned_workspace_store_is_rejected_before_runtime_allocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = SQLiteStore(workspace / "caller-owned.sqlite")
    substrate = LocalResourceProviderSubstrate(workspace)
    builder = RuntimeBuilder.configured(Runtime, substrate=substrate)
    allocated = False
    leases_before = _sqlite_identity_lease_inventory()

    def unexpected_allocation(_builder: RuntimeBuilder[Runtime]) -> Runtime:
        nonlocal allocated
        allocated = True
        raise AssertionError("Runtime allocation crossed the store isolation boundary")

    monkeypatch.setattr(RuntimeBuilder, "_allocate_host", unexpected_allocation)
    try:
        with pytest.raises(
            ValidationError,
            match="persistent SQLite runtime store must be outside",
        ):
            builder.from_store(store)
        assert not allocated
        assert store.conn.execute("SELECT 1").fetchone()[0] == 1
        assert _sqlite_identity_lease_inventory() == leases_before
    finally:
        store.close()


def test_async_caller_owned_workspace_store_is_rejected_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = SQLiteStore(workspace / "caller-owned-async.sqlite")
    substrate = LocalResourceProviderSubstrate(workspace)
    builder = RuntimeBuilder.configured(Runtime, substrate=substrate)
    allocated = False
    leases_before = _sqlite_identity_lease_inventory()

    def unexpected_allocation(_builder: RuntimeBuilder[Runtime]) -> Runtime:
        nonlocal allocated
        allocated = True
        raise AssertionError("Runtime allocation crossed the store isolation boundary")

    monkeypatch.setattr(RuntimeBuilder, "_allocate_host", unexpected_allocation)
    async def exercise() -> None:
        with pytest.raises(
            ValidationError,
            match="persistent SQLite runtime store must be outside",
        ):
            await builder.afrom_store(store)

    try:
        asyncio.run(exercise())
        assert not allocated
        assert store.conn.execute("SELECT 1").fetchone()[0] == 1
        assert _sqlite_identity_lease_inventory() == leases_before
    finally:
        store.close()


@pytest.mark.parametrize("async_attach", (False, True), ids=("sync", "async"))
def test_factory_opened_caller_owned_workspace_alias_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    async_attach: bool,
) -> None:
    workspace = tmp_path / "workspace"
    external = tmp_path / "external"
    workspace.mkdir()
    external.mkdir()
    target = external / "runtime.sqlite"
    alias = workspace / "runtime-alias.sqlite"
    try:
        alias.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"host cannot create file symlinks: {exc}")
    store = open_store(alias)
    assert isinstance(store, SQLiteStore)
    builder = RuntimeBuilder.configured(Runtime)
    monkeypatch.chdir(workspace)
    allocated = False
    leases_before = _sqlite_identity_lease_inventory()

    def unexpected_allocation(_builder: RuntimeBuilder[Runtime]) -> Runtime:
        nonlocal allocated
        allocated = True
        raise AssertionError("Runtime allocation crossed the store isolation boundary")

    monkeypatch.setattr(RuntimeBuilder, "_allocate_host", unexpected_allocation)
    try:
        if async_attach:
            async def exercise() -> None:
                with pytest.raises(
                    ValidationError,
                    match="persistent SQLite runtime store must be outside",
                ):
                    await builder.afrom_store(store)

            asyncio.run(exercise())
        else:
            with pytest.raises(
                ValidationError,
                match="persistent SQLite runtime store must be outside",
            ):
                builder.from_store(store)
        assert not allocated
        assert store.conn.execute("SELECT 1").fetchone()[0] == 1
        assert _sqlite_identity_lease_inventory() == leases_before
    finally:
        store.close()


def test_external_sqlite_and_explicit_memory_targets_remain_supported(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    state = tmp_path / "state"
    workspace.mkdir()
    state.mkdir()
    substrate = LocalResourceProviderSubstrate(workspace)
    external = state / "runtime.sqlite"

    runtime = Runtime.open(external, substrate=substrate)
    try:
        assert Path(runtime.store.path) == external
        assert external.exists()
    finally:
        runtime.close()

    first = Runtime.open("local", substrate=substrate)
    try:
        record = first.audit.record(actor="test:memory", action="test.memory.first")
        assert first.store.path == ":memory:"
    finally:
        first.close()
    second = Runtime.open(":memory:", substrate=substrate)
    try:
        assert second.store.path == ":memory:"
        assert all(item.record_id != record.record_id for item in second.audit.trace())
    finally:
        second.close()
    third = Runtime.open("sqlite://", substrate=substrate)
    try:
        assert third.store.path == ":memory:"
        assert all(item.record_id != record.record_id for item in third.audit.trace())
    finally:
        third.close()


def test_user_target_display_is_resolved_without_filesystem_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    _set_user_home(monkeypatch, home)
    expected = home / ".agent-libos" / "runtime" / "agent-libos.sqlite"

    resolved = resolve_store_target("user")

    assert resolved.backend == "sqlite"
    assert resolved.connection_target == str(expected)
    assert resolved.uses_user_directory
    assert display_store_target("user") == str(expected)
    assert not (home / ".agent-libos").exists()


def test_resolved_user_store_keeps_home_and_secure_directory_frozen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_home = tmp_path / "original-home"
    changed_home = tmp_path / "changed-home"
    original_home.mkdir()
    changed_home.mkdir()
    _set_user_home(monkeypatch, original_home)
    resolved = resolve_store_target("user")
    _set_user_home(monkeypatch, changed_home)

    store = open_store(resolved)
    try:
        expected_directory = original_home / ".agent-libos" / "runtime"
        assert Path(store.path) == expected_directory / "agent-libos.sqlite"
        assert (original_home / ".agent-libos").is_dir()
        assert expected_directory.is_dir()
        if os.name == "posix":
            assert (
                stat.S_IMODE(os.stat(original_home / ".agent-libos").st_mode)
                == 0o700
            )
            assert stat.S_IMODE(os.stat(expected_directory).st_mode) == 0o700
        assert not (changed_home / ".agent-libos").exists()
    finally:
        store.close()


def test_runtime_aopen_uses_default_persistent_user_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    home.mkdir()
    workspace.mkdir()
    _set_user_home(monkeypatch, home)
    monkeypatch.chdir(workspace)
    expected = home / ".agent-libos" / "runtime" / "agent-libos.sqlite"

    async def exercise() -> None:
        runtime = await Runtime.aopen()
        try:
            record = runtime.audit.record(
                actor="test:async-user-store",
                action="test.async_user_store.persisted",
            )
        finally:
            await runtime.ashutdown()
        reopened = await Runtime.aopen()
        try:
            assert Path(reopened.store.path) == expected
            assert any(
                item.record_id == record.record_id
                for item in reopened.audit.trace(actor="test:async-user-store")
            )
        finally:
            await reopened.ashutdown()

    asyncio.run(exercise())


def test_user_store_directory_symlink_is_rejected_without_database_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    redirected = tmp_path / "redirected"
    home.mkdir()
    redirected.mkdir()
    try:
        (home / ".agent-libos").symlink_to(redirected, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"host cannot create directory symlinks: {exc}")
    _set_user_home(monkeypatch, home)
    leases_before = _sqlite_identity_lease_inventory()

    with pytest.raises(ValidationError, match="symlink or unsafe component"):
        open_store("user")

    assert not (redirected / "runtime").exists()
    assert _sqlite_artifacts(redirected) == ()
    assert _sqlite_identity_lease_inventory() == leases_before


def test_user_store_symlink_is_rejected_without_o_nofollow_support(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    redirected = tmp_path / "redirected"
    home.mkdir()
    redirected.mkdir()
    try:
        (home / ".agent-libos").symlink_to(redirected, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"host cannot create directory symlinks: {exc}")
    _set_user_home(monkeypatch, home)
    monkeypatch.delattr("agent_libos.storage.factory.os.O_NOFOLLOW", raising=False)
    leases_before = _sqlite_identity_lease_inventory()

    with pytest.raises(ValidationError, match="symlink or unsafe component"):
        open_store("user")

    assert not (redirected / "runtime").exists()
    assert _sqlite_identity_lease_inventory() == leases_before


def test_user_store_database_leaf_symlink_is_rejected_without_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    runtime_directory = home / ".agent-libos" / "runtime"
    redirected = tmp_path / "redirected.sqlite"
    runtime_directory.mkdir(parents=True, mode=0o700)
    alias = runtime_directory / "agent-libos.sqlite"
    try:
        alias.symlink_to(redirected)
    except OSError as exc:
        pytest.skip(f"host cannot create file symlinks: {exc}")
    _set_user_home(monkeypatch, home)
    leases_before = _sqlite_identity_lease_inventory()

    with pytest.raises(ValidationError, match="symlink or unsafe component"):
        open_store("user")

    assert alias.is_symlink()
    assert not redirected.exists()
    assert _sqlite_identity_lease_inventory() == leases_before


def test_default_user_store_inside_workspace_is_rejected_before_directory_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _set_user_home(monkeypatch, workspace)
    monkeypatch.chdir(workspace)
    leases_before = _sqlite_identity_lease_inventory()

    with pytest.raises(
        ValidationError,
        match="persistent SQLite runtime store must be outside",
    ):
        Runtime.open()

    assert not (workspace / ".agent-libos").exists()
    assert _sqlite_identity_lease_inventory() == leases_before


def test_non_local_substrate_does_not_apply_local_workspace_store_boundary(
    tmp_path: Path,
) -> None:
    target = tmp_path / "remote-substrate-state.sqlite"
    builder = RuntimeBuilder.configured(
        Runtime,
        substrate=object(),  # type: ignore[arg-type]
    )

    resolved = builder._resolve_target_store_workspace_isolation(target)

    assert resolved.persistent_sqlite_path == target
    assert not target.exists()


def test_postgres_targets_are_not_classified_as_local_sqlite_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dsn = "postgresql://agent:secret@localhost/agent_libos"
    opened: list[tuple[str, bool]] = []

    class FakePostgresStore:
        def __init__(
            self,
            target: str,
            *,
            config: AgentLibOSConfig,
            initialize_schema: bool,
        ) -> None:
            del config
            opened.append((target, initialize_schema))

    monkeypatch.setattr(
        "agent_libos.storage.postgres.PostgresStore",
        FakePostgresStore,
    )
    resolved = resolve_store_target(dsn)
    store = open_store(dsn)

    assert resolved.backend == "postgres"
    assert resolved.persistent_sqlite_path is None
    assert resolved.display_target == "postgresql://***@localhost/agent_libos"
    assert isinstance(store, FakePostgresStore)
    assert opened == [(dsn, True)]


def test_windows_legacy_store_scans_bracket_durable_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    original_validate_payloads = TaskRunManager.validate_recoverable_payloads
    original_recover_startup = TaskRunManager.recover_startup

    def scan(**_kwargs: object) -> WindowsStoreIdentityValidationSummary:
        order.append("windows")
        return WindowsStoreIdentityValidationSummary(platform_checked=True)

    def validate_payloads(manager: TaskRunManager) -> object:
        order.append("payload")
        return original_validate_payloads(manager)

    def recover_startup(manager: TaskRunManager) -> object:
        order.append("task-run-recovery")
        return original_recover_startup(manager)

    monkeypatch.setattr(
        "agent_libos.runtime.builder.validate_legacy_windows_store_identities",
        scan,
    )
    monkeypatch.setattr(
        TaskRunManager,
        "validate_recoverable_payloads",
        validate_payloads,
    )
    monkeypatch.setattr(TaskRunManager, "recover_startup", recover_startup)

    runtime = Runtime.open("local")
    try:
        assert order[0:2] == ["windows", "payload"]
        assert order[-2:] == ["task-run-recovery", "windows"]
        assert runtime.windows_store_identity_preflight.platform_checked
        assert runtime.windows_store_identity_post_recovery.platform_checked
    finally:
        runtime.close()


def test_windows_legacy_store_preflight_failure_precedes_starting_and_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Runtime] = {}
    observed_states: list[str] = []
    payload_validation_called = False
    original_recovery = RuntimeBuilder._recover_runtime_state

    def tracked_recovery(host: Runtime) -> None:
        captured["host"] = host
        original_recovery(host)

    def fail_scan(**_kwargs: object) -> WindowsStoreIdentityValidationSummary:
        observed_states.append(captured["host"].lifecycle.state)
        raise LegacyWindowsStoreIdentityError("injected non-canonical legacy state")

    def unexpected_payload_validation(_manager: TaskRunManager) -> None:
        nonlocal payload_validation_called
        payload_validation_called = True

    monkeypatch.setattr(
        RuntimeBuilder,
        "_recover_runtime_state",
        staticmethod(tracked_recovery),
    )
    monkeypatch.setattr(
        "agent_libos.runtime.builder.validate_legacy_windows_store_identities",
        fail_scan,
    )
    monkeypatch.setattr(
        TaskRunManager,
        "validate_recoverable_payloads",
        unexpected_payload_validation,
    )

    with pytest.raises(
        LegacyWindowsStoreIdentityError,
        match="injected non-canonical legacy state",
    ):
        Runtime.open("local")

    assert observed_states == ["recovering"]
    assert not payload_validation_called


def test_non_windows_legacy_store_validation_is_zero_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnexpectedReader:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"unexpected persisted-state read: {name}")

    monkeypatch.setattr(
        "agent_libos.runtime.windows_store_identity.os.name",
        "posix",
    )
    summary = validate_legacy_windows_store_identities(
        authority=UnexpectedReader(),  # type: ignore[arg-type]
        checkpoints=UnexpectedReader(),  # type: ignore[arg-type]
        filesystem=UnexpectedReader(),  # type: ignore[arg-type]
    )

    assert summary == WindowsStoreIdentityValidationSummary(platform_checked=False)
