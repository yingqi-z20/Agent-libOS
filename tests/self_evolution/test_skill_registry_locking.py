from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import pytest
from pydantic import BaseModel

from agent_libos import Runtime
from agent_libos.tools.base import SyncAgentTool, ToolContext
from tests.support.skills import write_skill_package


class _EmptyArgs(BaseModel):
    pass


class _ConcurrentRegistrationTool(SyncAgentTool[_EmptyArgs]):
    name = "concurrent_registration_tool"
    description = "Tool used to exercise concurrent registry publication."
    args_schema = _EmptyArgs

    def run(self, args: _EmptyArgs, ctx: ToolContext) -> dict[str, bool]:
        return {"ok": True}


class _ObservedLifecycleLock:
    def __init__(self, delegate: Any, activation_attempted: threading.Event) -> None:
        self._delegate = delegate
        self._activation_attempted = activation_attempted

    def __enter__(self) -> "_ObservedLifecycleLock":
        if threading.current_thread().name == "skill-activation-lock-order":
            self._activation_attempted.set()
        self._delegate.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._delegate.release()


def test_skill_activation_acquires_registry_lifecycle_before_store(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    skill_dir = write_skill_package(tmp_path, "registry-lock-skill", allowed_tools=["echo"])
    runtime = Runtime.open("local")
    registration_holds_lifecycle = threading.Event()
    release_registration = threading.Event()
    activation_attempted_lifecycle = threading.Event()
    activation_entered_store = threading.Event()
    errors: list[tuple[str, BaseException]] = []
    threads: list[threading.Thread] = []
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="verify registry lock ordering")
        runtime.skills.register_skill_from_path(skill_dir, actor="test", require_capability=False)

        observed_lifecycle_lock = _ObservedLifecycleLock(
            runtime._registry_lifecycle_lock,
            activation_attempted_lifecycle,
        )
        runtime._registry_lifecycle_lock = observed_lifecycle_lock
        monkeypatch.setattr(
            runtime.skills,
            "_lifecycle_lock",
            observed_lifecycle_lock,
        )
        monkeypatch.setattr(
            runtime.tools,
            "_registry_lifecycle_lock_value",
            observed_lifecycle_lock,
        )
        original_register_locked = runtime.tools._register_tool_locked
        original_transaction = runtime.store.transaction

        def pause_registration_with_lifecycle_held(*args: Any, **kwargs: Any) -> Any:
            registration_holds_lifecycle.set()
            if not release_registration.wait(timeout=5):
                raise AssertionError("registration was not released by the test")
            return original_register_locked(*args, **kwargs)

        @contextmanager
        def observe_transaction(*args: Any, **kwargs: Any) -> Any:
            with original_transaction(*args, **kwargs) as cur:
                if threading.current_thread().name == "skill-activation-lock-order":
                    activation_entered_store.set()
                yield cur

        monkeypatch.setattr(runtime.tools, "_register_tool_locked", pause_registration_with_lifecycle_held)
        monkeypatch.setattr(runtime.store, "transaction", observe_transaction)

        def capture(label: str, operation: Callable[[], Any]) -> None:
            try:
                operation()
            except BaseException as exc:
                errors.append((label, exc))

        registration = threading.Thread(
            name="tool-registration-lock-order",
            target=capture,
            args=("registration", lambda: runtime.tools.register_tool(_ConcurrentRegistrationTool())),
            daemon=True,
        )
        activation = threading.Thread(
            name="skill-activation-lock-order",
            target=capture,
            args=(
                "activation",
                lambda: runtime.skills.activate_skill(
                    pid,
                    "registry-lock-skill",
                    actor=pid,
                    require_capability=False,
                ),
            ),
            daemon=True,
        )
        threads = [registration, activation]

        registration.start()
        assert registration_holds_lifecycle.wait(timeout=2)
        activation.start()
        assert activation_attempted_lifecycle.wait(timeout=2)

        # At this point registration owns the lifecycle lock. Activation must
        # be waiting for it without already owning the store lock.
        entered_store_before_lifecycle = activation_entered_store.is_set()
        release_registration.set()
        for thread in threads:
            thread.join(timeout=3)

        assert not entered_store_before_lifecycle
        assert not [thread.name for thread in threads if thread.is_alive()]
        assert errors == []
        assert "registry-lock-skill" in runtime.process.get(pid).loaded_skills
        assert runtime.tools.resolve("concurrent_registration_tool").name == "concurrent_registration_tool"
    finally:
        release_registration.set()
        for thread in threads:
            thread.join(timeout=0.1)
        if not any(thread.is_alive() for thread in threads):
            runtime.close()


def test_skill_activation_failure_releases_registry_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_dir = write_skill_package(tmp_path, "registry-lock-failure", allowed_tools=["echo"])
    runtime = Runtime.open("local")
    registration_finished = threading.Event()
    registration_errors: list[BaseException] = []
    thread: threading.Thread | None = None
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="verify failed activation unlocks")
        runtime.skills.register_skill_from_path(skill_dir, actor="test", require_capability=False)
        original_get_skill = runtime.skills._get_skill

        def fail_after_skill_read(skill_id: str) -> Any:
            original_get_skill(skill_id)
            raise RuntimeError("injected activation preflight failure")

        monkeypatch.setattr(runtime.skills, "_get_skill", fail_after_skill_read)
        with pytest.raises(RuntimeError, match="injected activation preflight failure"):
            runtime.skills.activate_skill(
                pid,
                "registry-lock-failure",
                actor=pid,
                require_capability=False,
            )

        def register_after_failure() -> None:
            try:
                runtime.tools.register_tool(_ConcurrentRegistrationTool())
            except BaseException as exc:
                registration_errors.append(exc)
            finally:
                registration_finished.set()

        thread = threading.Thread(target=register_after_failure, daemon=True)
        thread.start()
        assert registration_finished.wait(timeout=2)
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert registration_errors == []
        assert runtime.tools.resolve("concurrent_registration_tool").name == "concurrent_registration_tool"
    finally:
        if thread is not None:
            thread.join(timeout=0.1)
        if thread is None or not thread.is_alive():
            runtime.close()


def test_skill_package_replace_waits_for_hash_pinned_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_id = "registry-cas-skill"
    skill_dir = write_skill_package(
        tmp_path,
        skill_id,
        allowed_tools=["echo"],
        body="# registry-cas-skill\n\nUse package A.\n",
    )
    runtime = Runtime.open("local")
    activation_at_publication = threading.Event()
    release_activation = threading.Event()
    replacement_attempted = threading.Event()
    replacement_entered_store = threading.Event()
    errors: list[tuple[str, BaseException]] = []
    results: dict[str, Any] = {}
    threads: list[threading.Thread] = []
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="linearize Skill activation with package replacement",
        )
        package_a = runtime.skills.register_skill_from_path(
            skill_dir,
            actor="test.host",
            require_capability=False,
        )
        write_skill_package(
            tmp_path,
            skill_id,
            allowed_tools=["echo"],
            body="# registry-cas-skill\n\nUse package B.\n",
        )

        original_prepare = runtime.skills._prepare_jit_tools
        original_upsert = runtime.skills.store.upsert_skill

        def pause_after_hash_check(*args: Any, **kwargs: Any) -> Any:
            activation_at_publication.set()
            if not release_activation.wait(timeout=5):
                raise AssertionError("activation publication was not released")
            return original_prepare(*args, **kwargs)

        def observe_replacement_store(*args: Any, **kwargs: Any) -> Any:
            if threading.current_thread().name == "skill-package-replacement":
                replacement_entered_store.set()
            return original_upsert(*args, **kwargs)

        monkeypatch.setattr(runtime.skills, "_prepare_jit_tools", pause_after_hash_check)
        monkeypatch.setattr(runtime.skills.store, "upsert_skill", observe_replacement_store)

        def capture(label: str, operation: Callable[[], Any]) -> None:
            try:
                results[label] = operation()
            except BaseException as exc:
                errors.append((label, exc))

        activation = threading.Thread(
            name="skill-hash-pinned-activation",
            target=capture,
            args=(
                "activation",
                lambda: runtime.skills.activate_skill(
                    pid,
                    skill_id,
                    actor=pid,
                    require_capability=False,
                    expected_package_sha256=package_a["package_sha256"],
                ),
            ),
            daemon=True,
        )

        def replace_package() -> Any:
            replacement_attempted.set()
            return runtime.skills.register_skill_from_path(
                skill_dir,
                actor="test.host",
                replace=True,
                require_capability=False,
            )

        replacement = threading.Thread(
            name="skill-package-replacement",
            target=capture,
            args=("replacement", replace_package),
            daemon=True,
        )
        threads = [activation, replacement]

        activation.start()
        assert activation_at_publication.wait(timeout=2)
        replacement.start()
        assert replacement_attempted.wait(timeout=2)
        assert not replacement_entered_store.wait(timeout=0.2)

        release_activation.set()
        for thread in threads:
            thread.join(timeout=3)

        assert not [thread.name for thread in threads if thread.is_alive()]
        assert errors == []
        assert results["activation"]["package_sha256"] == package_a["package_sha256"]
        assert results["replacement"]["package_sha256"] != package_a["package_sha256"]
        loaded = runtime.process.get(pid).loaded_skills[skill_id]
        assert loaded["package_sha256"] == package_a["package_sha256"]
        discovered = runtime.skills.discover_skills(
            skill_id,
            actor=pid,
            require_capability=False,
            limit=1,
        )
        assert discovered[0]["package_sha256"] == results["replacement"]["package_sha256"]
        assert discovered[0]["active"] is False
    finally:
        release_activation.set()
        for thread in threads:
            thread.join(timeout=0.1)
        if not any(thread.is_alive() for thread in threads):
            runtime.close()
