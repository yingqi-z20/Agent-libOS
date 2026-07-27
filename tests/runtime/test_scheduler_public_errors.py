from __future__ import annotations

from agent_libos import Runtime
from agent_libos.utils.serde import dumps, to_jsonable


def test_scheduler_failure_result_and_process_status_hide_internal_exception() -> None:
    runtime = Runtime.open("local")
    secret = "R8m2Kx7Qp4Vn9Lc5Wd3Hj6Ta"
    path = "/Users/private/runtime.sqlite"
    sql = "SELECT * FROM provider_credentials"
    failure = RuntimeError(
        f"driver failed at {path}; opaque={secret}; SQL={sql}"
    )
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="scheduler failure")

        def fail_quantum(_pid: str) -> None:
            raise failure

        results = runtime.scheduler.run_pid_until_idle(
            pid,
            fail_quantum,
            max_quanta=1,
        )

        assert len(results) == 1
        result = results[0]
        assert result["ok"] is False
        assert result["code"] == "internal_error"
        assert result["error_type"] == "RuntimeError"
        assert result["correlation_id"].startswith("corr_")
        process = runtime.process.get(pid)
        assert result["correlation_id"] in (process.status_message or "")

        outward = dumps({"result": result, "process": to_jsonable(process)})
        assert secret not in outward
        assert path not in outward
        assert sql not in outward

        audit = next(
            record
            for record in runtime.audit.trace()
            if record.action == "scheduler.process_task_failed"
        )
        assert audit.correlation_id == result["correlation_id"]
        internal = dumps(audit.decision)
        assert secret not in internal
        assert path not in internal
        assert sql not in internal
        observation = audit.decision["internal_error"]
        assert observation["exception_text"]["bytes"] > 0
        assert len(observation["exception_text"]["sha256"]) == 64
    finally:
        runtime.close()
