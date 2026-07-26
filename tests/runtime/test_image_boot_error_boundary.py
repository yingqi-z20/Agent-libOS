from __future__ import annotations

from pathlib import Path

import pytest

from agent_libos import Runtime
from agent_libos.models import ProcessStatus


def test_spawn_boot_failure_text_is_not_persisted() -> None:
    runtime = Runtime.open("local")
    secret = "SECRET /private/image/provider-credential"
    try:
        pid = runtime.process.spawn(goal="image boot error boundary")

        runtime.image_boot._fail_boot(
            pid,
            runtime.config.runtime.default_image_id,
            RuntimeError(secret),
            phase="test.injected",
        )

        process = runtime.process.get(pid)
        assert process.status is ProcessStatus.FAILED
        assert process.status_message is not None
        assert process.status_message.startswith("image_boot_failed: RuntimeError ")
        failed = [
            record
            for record in runtime.audit.trace(target=f"process:{pid}")
            if record.action == "image.boot.failed"
        ][-1]
        assert failed.correlation_id is not None
        assert failed.correlation_id in process.status_message
        assert failed.decision["error"]["correlation_id"] == failed.correlation_id
        assert len(
            failed.decision["internal_error"]["exception_text"]["sha256"]
        ) == 64
        assert secret not in f"{process!r} {failed!r}"
    finally:
        runtime.close()


@pytest.mark.parametrize("strict", [False, True])
def test_image_workspace_cleanup_failure_audit_is_text_free(
    monkeypatch: pytest.MonkeyPatch,
    strict: bool,
) -> None:
    runtime = Runtime.open("local")
    secret = "SECRET /private/image/workspace-token"
    installer = runtime.image_package_installer
    relative_root = (
        Path(runtime.config.image.materialized_workspace_root)
        / "pid_test"
        / "publication_test"
        / "image_test"
        / "workspace"
    ).as_posix()

    def fail_remove(_path: object) -> None:
        raise OSError(secret)

    try:
        monkeypatch.setattr("agent_libos.runtime.image_package.shutil.rmtree", fail_remove)
        if strict:
            with pytest.raises(OSError) as caught:
                installer._cleanup_materialized_workspace(
                    relative_root,
                    actor="runtime",
                    reason="test",
                    strict=True,
                )
            assert caught.value.args == (secret,)
        else:
            installer._cleanup_materialized_workspace(
                relative_root,
                actor="runtime",
                reason="test",
                strict=False,
            )

        failure = [
            record
            for record in runtime.audit.trace()
            if record.action == "image.workspace.cleanup_failed"
        ][-1]
        assert failure.correlation_id is not None
        assert failure.decision["error"]["correlation_id"] == failure.correlation_id
        assert failure.decision["error"]["code"] == "image_workspace_cleanup_failed"
        assert len(
            failure.decision["internal_error"]["exception_text"]["sha256"]
        ) == 64
        assert secret not in repr(failure)
    finally:
        runtime.close()
