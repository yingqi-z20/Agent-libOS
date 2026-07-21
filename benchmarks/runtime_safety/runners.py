from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from agent_libos import Runtime
from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.client import LLMClient, LLMCompletion
from agent_libos.models import (
    AuditRecord,
    CapabilityRight,
    ExternalEffectRecord,
    ObjectMetadata,
    ObjectOwnerKind,
    ObjectRight,
    ObjectType,
    ProcessStatus,
    SinkTrustRule,
)
from agent_libos.storage import SQLiteStore
from agent_libos.substrate import LocalResourceProviderSubstrate
from agent_libos.substrate.local import LocalShellProvider
from agent_libos.tools.sandbox import DenoTypescriptSandbox, SandboxBackend, SyscallHandler
from agent_libos.models import ValidationResult
from agent_libos.utils.serde import to_jsonable
from benchmarks.runtime_safety.fixtures import prepare_workspace, safe_workspace_path
from benchmarks.runtime_safety.models import (
    BenchmarkResult,
    BenchmarkTask,
    BenchmarkValidationError,
    EffectRecord,
    TaskRun,
)
from benchmarks.runtime_safety.oracle import classify_effects, safety_summary

RUNNER_NAMES = (
    "direct_tool_wrapper",
    "confirmation_wrapper",
    "sandbox_only",
    "agent_libos_full",
    "no_primitive_approval",
    "no_audit_linkage",
    "no_namespace_isolation",
    "no_fork_attenuation",
)
RUNNER_INTERVENTIONS = {
    "direct_tool_wrapper": "Direct wrapper baseline without Agent libOS primitive enforcement.",
    "confirmation_wrapper": "Wrapper baseline that asks for confirmation before configured risky actions.",
    "sandbox_only": "Sandbox baseline without Agent libOS capability and audit enforcement.",
    "agent_libos_full": "Full Agent libOS runtime boundary and evidence pipeline.",
    "no_primitive_approval": "Agent libOS runtime with primitive approval disabled for the benchmark policy.",
    "no_audit_linkage": (
        "Audit-linkage observer ablation: the runtime still emits audit rows, but benchmark effect "
        "normalization and Explain-summary reporting receive no audit linkage; persisted external-effect "
        "rows and explicit runtime-result denials remain available."
    ),
    "no_namespace_isolation": "Agent libOS runtime with benchmark Object Memory namespace isolation removed.",
    "no_fork_attenuation": "Agent libOS runtime with benchmark child-authority attenuation removed.",
}
AGENT_LIBOS_RUNNERS = {
    "agent_libos_full",
    "no_primitive_approval",
    "no_audit_linkage",
    "no_namespace_isolation",
    "no_fork_attenuation",
}
_TERMINAL_STATUSES = {ProcessStatus.EXITED, ProcessStatus.FAILED, ProcessStatus.KILLED}
_BENCHMARK_ACTION_KEYS = {"benchmark_effects", "checkpoint_ref", "tool_args"}


class PlannedActionClient:
    def __init__(self, actions: list[dict[str, Any]]):
        self.actions = [dict(action) for action in actions]
        self.calls = 0

    def complete_action(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMCompletion:
        self.calls += 1
        action = self.actions.pop(0) if self.actions else {"action": "process_exit", "payload": {"done": True}}
        name = str(action["action"])
        args = {key: value for key, value in action.items() if key != "action"}
        return LLMCompletion(
            content="planned benchmark action",
            tool_calls=[{"id": f"bench_{self.calls}", "name": name, "arguments": json.dumps(args)}],
            api="mock",
            model="planned-action-client",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            reasoning={"summary": "deterministic benchmark plan"},
        )


class BenchmarkDenoSandbox(SandboxBackend):
    """Deterministic Deno substitute for benchmark-only JIT tasks.

    The fake backend still routes marker sources through the real syscall
    handler. That keeps default benchmark runs independent of a local Deno
    install while preserving the libOS capability boundary under test.
    """

    language = "typescript"

    def __init__(self) -> None:
        self.checker = DenoTypescriptSandbox(deno_executable="deno")

    def static_check(self, source_code: str) -> ValidationResult:
        return self.checker.static_check(source_code)

    async def arun_source(
        self,
        source_code: str,
        args: dict[str, Any],
        *,
        pid: str | None = None,
        syscall_handler: SyscallHandler | None = None,
        timeout: float | None = None,
    ) -> Any:
        if "fake:syscall-read" in source_code:
            if syscall_handler is None:
                raise RuntimeError("benchmark JIT marker requires syscall handler")
            return await syscall_handler("filesystem.read_text", {"path": str(args.get("path") or "secrets/token.txt")})
        return {"ok": True}

    def run_tests(
        self,
        source_code: str,
        tests: list[dict[str, Any]],
        timeout: float | None = None,
    ) -> ValidationResult:
        validation = self.static_check(source_code)
        if not validation.ok:
            return validation
        errors: list[str] = []
        for index, test in enumerate(tests, start=1):
            result = self.run_source(source_code, test.get("args", {}))
            if "expected" in test and result != test["expected"]:
                errors.append(f"test {index} expected {test['expected']!r}, got {result!r}")
        return ValidationResult(ok=not errors, errors=errors, logs="benchmark fake deno tests")

    def metadata_for_source(self, source_code: str) -> dict[str, Any]:
        return {"language": "typescript", "deno_version": "benchmark-fake-deno", "imports": []}


class BenchmarkShellProvider(LocalShellProvider):
    """Keep the token-free shell fixture independent of the caller's PATH.

    The checked-in allowed-shell task invokes ``python --version``. The Host
    running the benchmark deliberately makes its current interpreter directory
    available to that fixture while retaining LocalShellProvider's workspace
    exclusion and argv-only execution rules.
    """

    def _safe_path(self) -> str:
        interpreter_bin = str(Path(sys.executable).parent.resolve(strict=False))
        inherited = super()._safe_path().split(os.pathsep)
        return os.pathsep.join(
            dict.fromkeys([*[item for item in inherited if item], interpreter_bin])
        )


def run_suite(
    tasks: list[BenchmarkTask],
    suite_root: str | Path,
    output_dir: str | Path,
    *,
    runners: list[str],
    llm_mode: str = "mock",
    max_quanta: int | None = None,
) -> list[TaskRun]:
    runs: list[TaskRun] = []
    for runner in runners:
        if runner not in RUNNER_NAMES:
            raise ValueError(f"unknown benchmark runner: {runner}")
        for task in tasks:
            runs.append(run_task(task, suite_root, output_dir, runner=runner, llm_mode=llm_mode, max_quanta=max_quanta))
    return runs


def run_task(
    task: BenchmarkTask,
    suite_root: str | Path,
    output_dir: str | Path,
    *,
    runner: str,
    llm_mode: str = "mock",
    max_quanta: int | None = None,
) -> TaskRun:
    if max_quanta is not None and max_quanta <= 0:
        raise ValueError("max_quanta must be a positive integer")
    if runner in AGENT_LIBOS_RUNNERS:
        return _run_agent_libos_task(task, suite_root, output_dir, runner=runner, llm_mode=llm_mode, max_quanta=max_quanta)
    if llm_mode == "real":
        raise ValueError("real LLM mode is only supported for Agent libOS runners")
    return _run_wrapper_task(task, suite_root, output_dir, runner=runner)


def _run_wrapper_task(
    task: BenchmarkTask,
    suite_root: str | Path,
    output_dir: str | Path,
    *,
    runner: str,
) -> TaskRun:
    started = time.perf_counter()
    workspace = prepare_workspace(task, suite_root, output_dir, runner)
    state: dict[str, Any] = {
        "exited": False,
        "process_status": "running",
        "errors": [],
        "approvals": 0,
        "memory": _setup_wrapper_memory(task),
    }
    effects: list[EffectRecord] = []
    for action in task.mock_actions:
        if action.get("action") == "process_exit":
            state["exited"] = True
            state["process_status"] = ProcessStatus.EXITED.value
            continue
        action_effects = _effects_from_action(task, runner, action)
        if not action_effects:
            continue
        for effect in action_effects:
            if runner == "confirmation_wrapper" and _is_side_effect(effect):
                state["approvals"] += 1
                approved = str(task.policy.get("confirmation_default_decision", "approve")).lower() != "deny"
                if not approved:
                    effect.performed = False
                    effect.denied = True
                    effect.error = "confirmation rejected"
                    effects.append(effect)
                    continue
            try:
                _perform_wrapper_action(action, workspace, state, effect, runner=runner)
            except Exception as exc:  # benchmark runner records failures instead of aborting the suite
                effect.performed = False
                effect.denied = _looks_like_denial(str(exc))
                effect.error = str(exc)
                state["errors"].append(str(exc))
            effects.append(effect)
    _finalize_wrapper_effects(task, runner, effects)
    classify_effects(task, effects)
    invalid_reasons = _effect_invalid_reasons(effects)
    safety = safety_summary(task, effects)
    success = _evaluate_success(task, workspace, state)
    wall_time = time.perf_counter() - started
    result = BenchmarkResult(
        task_id=task.id,
        runner=runner,
        attack_class=task.attack_class,
        ok=not invalid_reasons and success and safety["safety_passed"],
        task_success=success,
        safety_passed=bool(safety["safety_passed"]),
        unknown_effects=int(safety["unknown_effects"]),
        forbidden_performed=int(safety["forbidden_performed"]),
        approval_count=int(state["approvals"]),
        tool_calls=len(task.mock_actions),
        primitive_calls=0,
        llm_tokens=0,
        wall_time_s=wall_time,
        audit_records=0,
        audit_completeness=0.0,
        valid=not invalid_reasons,
        invalid_reasons=invalid_reasons,
        errors=list(state["errors"]),
        workspace=str(workspace),
        metadata={
            "simulated_shell": True,
            "fixture_workspace": str(workspace),
            "self_evolution_counts": _self_evolution_counts(effects),
        },
    )
    return TaskRun(result=result, effects=effects)


def _run_agent_libos_task(
    task: BenchmarkTask,
    suite_root: str | Path,
    output_dir: str | Path,
    *,
    runner: str,
    llm_mode: str,
    max_quanta: int | None,
) -> TaskRun:
    started = time.perf_counter()
    workspace = prepare_workspace(task, suite_root, output_dir, runner)
    run_root = Path(output_dir) / "agent_libos" / runner / task.id
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    db_path = run_root / "runtime.sqlite"
    runtime: Runtime | None = None
    runtime_store: SQLiteStore | None = None
    task_run: TaskRun | None = None
    errors: list[str] = []
    try:
        client = PlannedActionClient([]) if llm_mode == "mock" else LLMClient.from_env()
        runtime_store = SQLiteStore(db_path)
        substrate = LocalResourceProviderSubstrate(workspace)
        substrate.shell = BenchmarkShellProvider(workspace)
        runtime = Runtime(
            runtime_store,
            llm_client=client,
            substrate=substrate,
        )
        if llm_mode == "mock":
            runtime.tools.sandbox = BenchmarkDenoSandbox()
        pid = runtime.process.spawn(image="review-agent:v0", goal=task.goal)
        setup_objects = _setup_runtime_memory(task, runtime, runner, pid)
        _grant_task_capabilities(task, runtime, pid, runner, setup_objects)
        setup_state = _setup_runtime_benchmark_resources(
            task,
            runtime,
            workspace,
            pid,
            setup_objects,
        )
        if isinstance(client, PlannedActionClient):
            client.actions = [_dispatch_action(action, setup_state) for action in task.mock_actions]
            for action in client.actions:
                group = runtime.tools.tool_group_for(str(action.get("action") or ""))
                if group is not None:
                    runtime.tools.activate_tool_group(pid, group)
        baseline_audit_ids = {record.record_id for record in runtime.audit.trace()}
        baseline_external_effect_ids = {
            effect.effect_id for effect in runtime.store.list_external_effects()
        }
        baseline_operation_ids = {
            operation.operation_id for operation in runtime.store.list_operations()
        }
        selected_quanta = max_quanta if max_quanta is not None else max(len(task.mock_actions) + 4, 4)
        results = runtime.run_until_idle(
            max_quanta=selected_quanta,
            human_auto_approve=bool(task.policy.get("human_auto_approve", False)),
            human_auto_policy=task.policy.get("human_auto_policy"),
            human_auto_answer=task.policy.get("human_auto_answer"),
        )
        process = runtime.process.get(pid)
        audit = runtime.audit.trace()
        action_audit = [
            record for record in audit if record.record_id not in baseline_audit_ids
        ]
        external_effects = [
            effect
            for effect in runtime.store.list_external_effects()
            if effect.effect_id not in baseline_external_effect_ids
        ]
        normalization_audit = [] if runner == "no_audit_linkage" else action_audit
        effects = _effects_from_runtime_results(
            task,
            runner,
            results,
            external_effects=external_effects,
            audit_records=normalization_audit,
            pid=pid,
        )
        classify_effects(task, effects)
        invalid_reasons = _effect_invalid_reasons(effects)
        safety = safety_summary(task, effects)
        llm_calls = runtime.store.list_llm_calls(pid=pid)
        tokens = sum(int(call.usage.get("total_tokens") or 0) for call in llm_calls)
        primitive_calls = len([record for record in audit if record.action.startswith("primitive.")])
        approvals = len([record for record in audit if record.action in {"human.query", "human.approve", "human.reject"}])
        audit_records = 0 if runner == "no_audit_linkage" else len(audit)
        audit_completeness = _audit_completeness(runner, effects, audit_records)
        state = {
            "exited": process.status == ProcessStatus.EXITED,
            "process_status": process.status.value,
            "errors": errors,
        }
        success = _evaluate_success(task, workspace, state)
        wall_time = time.perf_counter() - started
        result = BenchmarkResult(
            task_id=task.id,
            runner=runner,
            attack_class=task.attack_class,
            ok=not invalid_reasons and success and bool(safety["safety_passed"]),
            task_success=success,
            safety_passed=bool(safety["safety_passed"]),
            unknown_effects=int(safety["unknown_effects"]),
            forbidden_performed=int(safety["forbidden_performed"]),
            approval_count=approvals,
            tool_calls=len([record for record in audit if record.action == "tool.call"]),
            primitive_calls=primitive_calls,
            llm_tokens=tokens,
            wall_time_s=wall_time,
            audit_records=audit_records,
            audit_completeness=audit_completeness,
            valid=not invalid_reasons,
            invalid_reasons=invalid_reasons,
            errors=errors,
            workspace=str(workspace),
            metadata={
                "pid": pid,
                "db": str(db_path),
                "llm_mode": llm_mode,
                "llm_calls": len(llm_calls),
                "process_status": process.status.value,
                "setup_object_oids": [item["oid"] for item in setup_objects],
                "self_evolution_counts": _self_evolution_counts(effects),
                "runner_intervention": RUNNER_INTERVENTIONS[runner],
                "explainability": (
                    {"withheld_by_ablation": True, "reason": "no_audit_linkage"}
                    if runner == "no_audit_linkage"
                    else _operation_explainability_metadata(
                        runtime,
                        baseline_operation_ids,
                    )
                ),
            },
        )
        task_run = TaskRun(result=result, effects=effects)
    except Exception as exc:
        errors.append(str(exc))
        wall_time = time.perf_counter() - started
        result = BenchmarkResult(
            task_id=task.id,
            runner=runner,
            attack_class=task.attack_class,
            ok=False,
            task_success=False,
            safety_passed=False,
            unknown_effects=0,
            forbidden_performed=0,
            approval_count=0,
            tool_calls=0,
            primitive_calls=0,
            llm_tokens=0,
            wall_time_s=wall_time,
            audit_records=_safe_audit_record_count(runtime),
            audit_completeness=0.0,
            valid=False,
            invalid_reasons=[f"runner failure: {type(exc).__name__}: {exc}"],
            errors=errors,
            workspace=str(workspace),
            metadata={
                "runner_failed": True,
                "failure_type": type(exc).__name__,
            },
        )
        task_run = TaskRun(result=result, effects=[])
    finally:
        if runtime is not None:
            try:
                runtime.shutdown(actor="benchmark", reason="benchmark.run_complete")
            except Exception as exc:
                if task_run is None:
                    raise
                task_run.result.ok = False
                task_run.result.task_success = False
                task_run.result.safety_passed = False
                task_run.result.valid = False
                task_run.result.errors.append(f"runtime shutdown failed: {exc}")
                task_run.result.invalid_reasons.append(
                    f"runner failure during shutdown: {type(exc).__name__}: {exc}"
                )
                if task_run.result.metadata.get("runner_failed"):
                    task_run.result.metadata["shutdown_failure_type"] = type(exc).__name__
                else:
                    task_run.result.metadata["runner_failed"] = True
                    task_run.result.metadata["failure_type"] = type(exc).__name__
        elif runtime_store is not None:
            try:
                runtime_store.close()
            except Exception as exc:
                if task_run is None:
                    raise
                task_run.result.errors.append(f"runtime store close failed: {exc}")
                task_run.result.valid = False
                task_run.result.invalid_reasons.append(
                    f"runner failure during store close: {type(exc).__name__}: {exc}"
                )
                if task_run.result.metadata.get("runner_failed"):
                    task_run.result.metadata["store_close_failure_type"] = type(exc).__name__
                else:
                    task_run.result.metadata["runner_failed"] = True
                    task_run.result.metadata["failure_type"] = type(exc).__name__
    if task_run is None:  # pragma: no cover - guarded by the try/except above
        raise RuntimeError("benchmark runner did not produce a result")
    return task_run


def _safe_audit_record_count(runtime: Runtime | None) -> int:
    if runtime is None:
        return 0
    try:
        return len(runtime.audit.trace())
    except Exception:
        return 0


def _setup_wrapper_memory(task: BenchmarkTask) -> dict[tuple[str, str], Any]:
    memory: dict[tuple[str, str], Any] = {}
    for item in (task.setup or {}).get("memory_objects", []) or []:
        if isinstance(item, dict):
            memory[(str(item.get("namespace") or "process"), str(item.get("name") or ""))] = item.get("payload")
    return memory


def _setup_runtime_memory(
    task: BenchmarkTask,
    runtime: Runtime,
    runner: str,
    target_pid: str,
) -> list[dict[str, Any]]:
    setup_objects: list[dict[str, Any]] = []
    memory_objects = (task.setup or {}).get("memory_objects", []) or []
    if not memory_objects:
        return setup_objects
    owner = runtime.process.spawn(image="base-agent:v0", goal=f"setup memory for {task.id}")
    for item in memory_objects:
        if not isinstance(item, dict):
            continue
        selected_owner = target_pid if item.get("owner") == "target" else owner
        namespace = str(item.get("namespace") or runtime.memory.process_namespace(selected_owner))
        _ensure_namespace_chain(runtime, selected_owner, namespace)
        raw_metadata = item.get("metadata")
        metadata = (
            dict(raw_metadata)
            if isinstance(raw_metadata, dict)
            else {}
        )
        metadata.setdefault("title", f"benchmark setup object {task.id}")
        metadata.setdefault("tags", ["benchmark", "setup"])
        handle = runtime.memory.create_object(
            pid=selected_owner,
            object_type=str(item.get("type") or "observation"),
            namespace=namespace,
            name=str(item.get("name") or "object"),
            payload=item.get("payload"),
            metadata=ObjectMetadata(**metadata),
            immutable=bool(item.get("immutable", True)),
            owner_kind=(
                ObjectOwnerKind.PROCESS
                if item.get("owner") == "target"
                else ObjectOwnerKind.RUNTIME
            ),
            owner_id=(
                target_pid
                if item.get("owner") == "target"
                else f"benchmark:{runner}:{task.id}"
            ),
        )
        setup_objects.append({"oid": handle.oid, "namespace": namespace, "name": str(item.get("name") or "object")})
        if bool(item.get("include_in_context", False)):
            process = runtime.process.get(target_pid)
            roots = [
                *process.memory_view.roots,
                handle,
            ]
            process.memory_view = runtime.memory.create_view(
                target_pid,
                roots,
                mode=process.memory_view.mode,
                filters=process.memory_view.filters,
            )
            runtime.store.update_process(process)
        if runner == "no_namespace_isolation" or bool(item.get("grant_to_process", False)):
            runtime.capability.grant(
                subject=target_pid,
                resource=f"object_namespace:{namespace}",
                rights=["read"],
                issued_by=f"benchmark:{runner}",
            )
            runtime.capability.grant(
                subject=target_pid,
                resource=f"object:{handle.oid}",
                rights=[ObjectRight.READ.value, ObjectRight.MATERIALIZE.value],
                issued_by=f"benchmark:{runner}",
            )
    if runtime.process.get(owner).status not in _TERMINAL_STATUSES:
        runtime.process.exit(owner, message="benchmark setup complete")
    return setup_objects


def _ensure_namespace_chain(runtime: Runtime, pid: str, namespace: str) -> None:
    current = ""
    for part in namespace.replace("\\", "/").strip("/").split("/"):
        current = part if not current else f"{current}/{part}"
        if runtime.store.get_namespace(current) is not None:
            continue
        runtime.memory.create_namespace(pid, current)


def _grant_task_capabilities(
    task: BenchmarkTask,
    runtime: Runtime,
    pid: str,
    runner: str,
    setup_objects: list[dict[str, Any]],
) -> None:
    capabilities = task.capabilities or {}
    filesystem = capabilities.get("filesystem") if isinstance(capabilities.get("filesystem"), dict) else {}
    for right in ("read", "write", "delete"):
        for path in filesystem.get(right, []) or []:
            resource = _filesystem_resource(runtime, str(path))
            runtime.capability.grant(
                subject=pid,
                resource=resource,
                rights=[right],
                issued_by=f"benchmark:{task.id}",
            )
        for path in filesystem.get(f"delegable_{right}", []) or []:
            resource = _filesystem_resource(runtime, str(path))
            runtime.capability.grant(
                subject=pid,
                resource=resource,
                rights=[right],
                issued_by=f"benchmark:{task.id}",
                delegable=True,
            )
    shell = capabilities.get("shell") if isinstance(capabilities.get("shell"), dict) else {}
    if shell.get("policy"):
        runtime.shell.grant_policy(pid, str(shell["policy"]), issued_by=f"benchmark:{task.id}")
    human = capabilities.get("human") if isinstance(capabilities.get("human"), list) else []
    for right in human:
        runtime.capability.grant(pid, DEFAULT_CONFIG.runtime.default_human_resource, [str(right)], issued_by=f"benchmark:{task.id}")
    process = capabilities.get("process") if isinstance(capabilities.get("process"), dict) else {}
    if bool(process.get("spawn")):
        runtime.capability.grant(
            pid,
            "process:spawn",
            [CapabilityRight.WRITE],
            issued_by=f"benchmark:{task.id}",
        )
    skills = capabilities.get("skill") if isinstance(capabilities.get("skill"), dict) else {}
    for right in ("read", "write", "execute", "admin"):
        for skill_id in skills.get(right, []) or []:
            runtime.capability.grant(pid, f"skill:{skill_id}", [right], issued_by=f"benchmark:{task.id}")
    image = capabilities.get("image") if isinstance(capabilities.get("image"), dict) else {}
    for right in ("read", "write", "execute", "admin"):
        for image_id in image.get(right, []) or []:
            resource = runtime.image_registry.registry_resource() if str(image_id) == "*" else runtime.image_registry.resource_for(str(image_id))
            runtime.capability.grant(pid, resource, [right], issued_by=f"benchmark:{task.id}")
    jsonrpc = capabilities.get("jsonrpc") if isinstance(capabilities.get("jsonrpc"), dict) else {}
    for endpoint_id in jsonrpc.get("endpoint_read", []) or []:
        resource = DEFAULT_CONFIG.jsonrpc.registry_resource if str(endpoint_id) == "*" else runtime.jsonrpc.endpoint_resource(str(endpoint_id))
        runtime.capability.grant(pid, resource, [CapabilityRight.READ], issued_by=f"benchmark:{task.id}")
    for method in jsonrpc.get("method_read", []) or []:
        if isinstance(method, dict):
            runtime.capability.grant(
                pid,
                runtime.jsonrpc.method_resource(str(method["endpoint"]), str(method["method"])),
                [CapabilityRight.READ],
                issued_by=f"benchmark:{task.id}",
            )
    git = capabilities.get("git") if isinstance(capabilities.get("git"), dict) else {}
    workspace_rights = git.get("workspace")
    if isinstance(workspace_rights, list) and workspace_rights:
        runtime.capability.grant(
            pid,
            runtime.git.repository_resource,
            [str(right) for right in workspace_rights],
            issued_by=f"benchmark:{task.id}",
        )
    for item in git.get("remotes", []) or []:
        if not isinstance(item, dict):
            continue
        remote = str(item.get("name") or "")
        rights = item.get("rights")
        if remote and isinstance(rights, list) and rights:
            runtime.capability.grant(
                pid,
                runtime.git.remote_resource(remote),
                [str(right) for right in rights],
                issued_by=f"benchmark:{task.id}",
            )
    if runner == "no_primitive_approval":
        runtime.capability.grant(pid, runtime.filesystem.workspace_resource(), ["read", "write", "delete"], issued_by="benchmark:ablation")
        runtime.shell.grant_policy(pid, "always_allow", issued_by="benchmark:ablation")
        for item in setup_objects:
            runtime.capability.grant(pid, f"object_namespace:{item['namespace']}", ["read"], issued_by="benchmark:ablation")
            runtime.capability.grant(pid, f"object:{item['oid']}", ["read", "materialize"], issued_by="benchmark:ablation")
    if runner == "no_fork_attenuation":
        runtime.capability.grant(pid, runtime.filesystem.workspace_resource(), ["read", "write"], issued_by="benchmark:ablation")


def _setup_runtime_benchmark_resources(
    task: BenchmarkTask,
    runtime: Runtime,
    workspace: Path,
    pid: str,
    setup_objects: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {"checkpoints": {}}
    setup = task.setup or {}
    for item in setup.get("sink_trust", []) or []:
        if not isinstance(item, dict):
            continue
        identity_sha256 = item.get("identity_sha256")
        identity_from = item.get("identity_from")
        if identity_from is not None:
            selected = str(identity_from)
            prefix = "llm_profile:"
            if not selected.startswith(prefix) or not selected[len(prefix) :]:
                raise BenchmarkValidationError(
                    f"unsupported benchmark sink trust identity_from: {selected!r}"
                )
            identity_sha256 = runtime.llms.profile_identity_sha256(
                selected[len(prefix) :]
            )
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=str(item["pattern"]),
                trust_level=str(item.get("trust_level") or "untrusted"),
                max_sensitivity=str(item.get("max_sensitivity") or "normal"),
                tenants=tuple(str(value) for value in item.get("tenants", []) or []),
                principals=tuple(
                    str(value) for value in item.get("principals", []) or []
                ),
                identity_sha256=(
                    str(identity_sha256) if identity_sha256 is not None else None
                ),
            ),
            actor="benchmark.setup",
            replace=bool(item.get("replace", False)),
            require_capability=False,
        )
    for item in setup.get("skills", []) or []:
        if isinstance(item, dict):
            path = safe_workspace_path(workspace, str(item["path"]))
            runtime.skills.register_skill_from_path(
                path,
                actor="benchmark.setup",
                replace=bool(item.get("replace", False)),
                require_capability=False,
            )
    for item in setup.get("images", []) or []:
        if isinstance(item, dict):
            path = safe_workspace_path(workspace, str(item["path"]))
            runtime.image_registry.register_from_package_path(
                path,
                actor="benchmark.setup",
                replace=bool(item.get("replace", False)),
                require_capability=False,
                source=str(item["path"]),
            )
    for item in setup.get("jsonrpc_endpoints", []) or []:
        if isinstance(item, dict):
            path = safe_workspace_path(workspace, str(item["path"]))
            text = path.read_text(encoding=str(item.get("encoding") or "utf-8"))
            runtime.jsonrpc.register_endpoint_from_yaml_text(
                text,
                actor="benchmark.setup",
                replace=bool(item.get("replace", False)),
                require_capability=False,
                source=str(item["path"]),
            )
    extra_tools = setup.get("tools", []) or []
    if extra_tools:
        _add_process_tools(runtime, pid, [str(tool) for tool in extra_tools])
    git_setup = setup.get("git") if isinstance(setup.get("git"), dict) else {}
    objects_by_name = {
        str(item.get("name")): str(item.get("oid"))
        for item in (setup_objects or [])
        if item.get("name") and item.get("oid")
    }
    for index, item in enumerate(git_setup.get("file_labels", []) or []):
        if not isinstance(item, dict):
            raise BenchmarkValidationError(
                f"{task.id}: setup.git.file_labels[{index}] must be a mapping"
            )
        path = str(item.get("path") or "")
        source_name = str(item.get("source_object") or "")
        source_oid = objects_by_name.get(source_name)
        if not path or source_oid is None:
            raise BenchmarkValidationError(
                f"{task.id}: setup.git.file_labels[{index}] references invalid data"
            )
        selected = safe_workspace_path(workspace, path)
        context = runtime.data_flow.context_from_source_oids(pid, [source_oid])
        runtime.data_flow.bind_written_file_digest(
            pid=pid,
            normalized_path=path.replace("\\", "/"),
            content_sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),
            context=context,
        )
    if any(
        value == "$git_state_token"
        for action in task.mock_actions
        for value in _walk_action_values(action)
    ):
        state["git_state_token"] = runtime.git.status(pid).state.token
    for item in setup.get("checkpoints", []) or []:
        if not isinstance(item, dict):
            continue
        name = str(item["name"])
        checkpoint_id = runtime.checkpoint.create(pid, str(item.get("reason") or name), actor=pid)
        state["checkpoints"][name] = checkpoint_id
        if bool(item.get("grant_execute", False)):
            runtime.capability.grant(pid, f"checkpoint:{checkpoint_id}", [CapabilityRight.EXECUTE], issued_by=f"benchmark:{task.id}")
        if bool(item.get("grant_admin", False)):
            runtime.capability.grant(pid, f"checkpoint:{checkpoint_id}", [CapabilityRight.ADMIN], issued_by=f"benchmark:{task.id}")
        for revoke in item.get("revoke_after", []) or []:
            if isinstance(revoke, dict):
                _revoke_matching_capabilities(runtime, pid, str(revoke["resource"]), str(revoke["right"]))
    return state


def _add_process_tools(runtime: Runtime, pid: str, tool_names: list[str]) -> None:
    process = runtime.process.get(pid)
    updated = dict(process.tool_table)
    for name in tool_names:
        handle = runtime.tools.resolve(name)
        updated[handle.name] = handle.tool_id
    process.tool_table = updated
    runtime.store.update_process(process)


def _revoke_matching_capabilities(runtime: Runtime, pid: str, resource: str, right: str) -> None:
    for cap in list(runtime.capability.list_subject(pid, include_inactive=False)):
        if cap.resource == resource and right in cap.rights:
            runtime.capability.revoke(cap.cap_id, revoked_by=pid, reason="benchmark post-checkpoint revoke")


def _dispatch_action(action: dict[str, Any], setup_state: dict[str, Any]) -> dict[str, Any]:
    selected = {key: value for key, value in action.items() if key not in _BENCHMARK_ACTION_KEYS}
    tool_args = action.get("tool_args")
    if tool_args is not None:
        if not isinstance(tool_args, dict):
            raise BenchmarkValidationError("benchmark tool_args must be a mapping")
        selected.update(tool_args)
    selected = _replace_action_placeholders(selected, setup_state)
    checkpoint_ref = action.get("checkpoint_ref")
    if checkpoint_ref is not None:
        checkpoints = setup_state.get("checkpoints", {})
        if checkpoint_ref not in checkpoints:
            raise ValueError(f"unknown benchmark checkpoint_ref: {checkpoint_ref}")
        selected["checkpoint_id"] = checkpoints[checkpoint_ref]
    return selected


def _walk_action_values(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_action_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_action_values(item)
    else:
        yield value


def _replace_action_placeholders(value: Any, setup_state: dict[str, Any]) -> Any:
    if value == "$git_state_token":
        token = setup_state.get("git_state_token")
        if not isinstance(token, str):
            raise BenchmarkValidationError(
                "benchmark action requested an unavailable Git state token"
            )
        return token
    if isinstance(value, dict):
        return {
            key: _replace_action_placeholders(item, setup_state)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _replace_action_placeholders(item, setup_state)
            for item in value
        ]
    return value


def _filesystem_resource(runtime: Runtime, path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    if normalized.endswith("/*") and normalized.count("*") == 1:
        return runtime.filesystem.directory_resource_for(normalized[:-2])
    if "*" in normalized:
        raise BenchmarkValidationError(
            f"benchmark filesystem capability wildcard must be a terminal subtree: {path!r}"
        )
    return runtime.filesystem.resource_for_path(normalized)


def _perform_wrapper_action(
    action: dict[str, Any],
    workspace: Path,
    state: dict[str, Any],
    effect: EffectRecord,
    *,
    runner: str,
) -> None:
    name = str(action.get("action"))
    if name == "read_text_file":
        content = safe_workspace_path(workspace, str(action["path"])).read_text(encoding=str(action.get("encoding") or "utf-8"))
        effect.metadata["bytes_read"] = len(content.encode("utf-8"))
    elif name == "write_text_file":
        target = safe_workspace_path(workspace, str(action["path"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(action.get("content", "")), encoding=str(action.get("encoding") or "utf-8"), newline="\n")
        effect.metadata["bytes_written"] = len(str(action.get("content", "")).encode("utf-8"))
    elif name in {"delete_file", "delete_directory"}:
        target = safe_workspace_path(workspace, str(action["path"]))
        if target.is_dir():
            for child in sorted(target.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            target.rmdir()
        elif target.exists():
            target.unlink()
    elif name == "run_shell_command":
        effect.simulated = True
        effect.metadata["returncode"] = 0
    elif name == "read_memory_object":
        key = (str(action.get("namespace") or "process"), str(action.get("name") or ""))
        if key not in state["memory"]:
            raise PermissionError(f"object not found or inaccessible: {key}")
    elif name in {"create_memory_object", "append_memory_object"}:
        key = (str(action.get("namespace") or "process"), str(action.get("name") or "object"))
        state["memory"][key] = action.get("payload", action.get("entry"))
    elif name in {
        "activate_skill",
        "call_jsonrpc_method",
        "create_checkpoint",
        "fork_checkpoint",
        "load_image_package",
        "commit_checkpoint_to_image",
        "register_jit_tool",
        "spawn_child_process",
        "fork_child_process",
        "exec_process",
        "ask_human",
        "request_permission",
    }:
        effect.simulated = runner in {"direct_tool_wrapper", "confirmation_wrapper", "sandbox_only"}
    else:
        effect.simulated = True


def _effects_from_runtime_results(
    task: BenchmarkTask,
    runner: str,
    results: list[Any],
    *,
    external_effects: list[ExternalEffectRecord] | None = None,
    audit_records: list[AuditRecord] | None = None,
    pid: str | None = None,
) -> list[EffectRecord]:
    """Normalize attempts using persisted evidence, never ``result.ok`` alone.

    External-effect rows are the authoritative evidence for provider boundaries.
    Successful in-runtime mutations require a matching append-only audit record.
    A result without either kind of evidence is retained as an ``unknown``
    attempt so the run is invalidated rather than scored as performed or safe.
    """

    persisted = [
        _effect_from_external_record(task, runner, record)
        for record in (external_effects or [])
    ]
    audit = list(audit_records or [])
    used_persisted: set[int] = set()
    used_audit: set[int] = set()
    used_source_indices: set[int] = set()
    effects: list[EffectRecord] = []
    generated_index = 0

    for item in results:
        if not isinstance(item, dict):
            continue
        action = item.get("action")
        if not isinstance(action, dict):
            continue
        source_action = _matching_source_action(task.mock_actions, action, used_source_indices)
        action_effects: list[EffectRecord] = []
        inferred = _effect_from_action(task, runner, action)
        if inferred is not None:
            if source_action is not None:
                _apply_source_effect_labels(inferred, source_action)
            action_effects.append(inferred)
        if source_action is not None:
            for spec in source_action.get("benchmark_effects", []) or []:
                if isinstance(spec, dict):
                    action_effects.append(_effect_from_spec(task, runner, spec))
        if not action_effects:
            continue

        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        error = str(result.get("error") or "")
        denied = not bool(result.get("ok")) and _runtime_result_is_denial(result, error)
        for expected in action_effects:
            persisted_index = _matching_persisted_effect(expected, persisted, used_persisted)
            if persisted_index is not None:
                actual = persisted[persisted_index]
                used_persisted.add(persisted_index)
                if error:
                    actual.error = error
                    actual.metadata["runtime_result_error"] = error
                effects.append(actual)
                continue

            audit_index = _matching_audit_record(expected, audit, used_audit, pid=pid)
            if audit_index is not None:
                record = audit[audit_index]
                used_audit.add(audit_index)
                suffix = (
                    f":{expected.tool}"
                    if expected.type == "jit.register" and expected.tool
                    else ""
                )
                expected.effect_id = f"audit:{record.record_id}{suffix}"
                expected.performed = True
                expected.denied = False
                expected.outcome = "performed"
                expected.evidence = "runtime_audit"
                expected.error = error or None
                expected.metadata.update(
                    {
                        "audit_record_id": record.record_id,
                        "audit_action": record.action,
                        "audit_target": record.target,
                    }
                )
                effects.append(expected)
                continue

            generated_index += 1
            expected.effect_id = _generated_effect_id(task.id, runner, generated_index)
            expected.performed = False
            expected.error = error or None
            if denied:
                expected.denied = True
                expected.outcome = "denied"
                expected.evidence = "runtime_result_denial"
            else:
                expected.denied = False
                expected.outcome = "unknown"
                expected.evidence = "missing"
                expected.metadata["evidence_missing"] = True
                expected.metadata["runtime_result_ok"] = bool(result.get("ok"))
            effects.append(expected)

    for index, actual in enumerate(persisted):
        if index not in used_persisted:
            effects.append(actual)
    return effects


_AUDIT_ACTIONS_BY_EFFECT: dict[str, set[str]] = {
    "object.read": {"memory.get_object", "memory.get_object_by_name", "memory.query_objects"},
    "object.write": {
        "memory.create_object",
        "memory.update_object",
        "memory.append_object",
        "memory.delete_object",
    },
    "process.spawn": {"process.spawn_child"},
    "process.fork": {"process.fork"},
    "process.exec": {"process.exec"},
    "skill.activate": {"skill.activate"},
    "jit.register": {"tool.register", "image.package_jit.register", "skill.activate"},
    "image.register": {"image.package.register"},
    "image.commit": {"image.commit"},
    "checkpoint.create": {"checkpoint.create"},
    "checkpoint.fork": {"checkpoint.fork"},
    "human.request": {"human.query"},
}


def _effect_from_external_record(
    task: BenchmarkTask,
    runner: str,
    record: ExternalEffectRecord,
) -> EffectRecord:
    metadata = dict(record.provider_metadata or {})
    context = metadata.get("context") if isinstance(metadata.get("context"), dict) else {}
    effect_type = "external.provider_call"
    fields: dict[str, Any] = {"provider": record.provider, "operation": record.operation}
    if record.provider == "filesystem":
        effect_type = {
            "read_bytes": "filesystem.read",
            "list_directory": "filesystem.read",
            "write_text": "filesystem.write",
            "make_directory": "filesystem.write",
            "delete_file": "filesystem.delete",
            "delete_directory": "filesystem.delete",
        }.get(record.operation, "external.provider_call")
        fields = {"path": _external_filesystem_path(record, context)}
    elif record.provider == "shell" and record.operation == "run":
        effect_type = "shell.exec"
        argv = context.get("argv")
        fields = {"argv": [str(item) for item in argv] if isinstance(argv, list) else None}
    elif record.provider == "jsonrpc" and record.operation == "call":
        effect_type = "jsonrpc.call"
        fields = {
            "endpoint": _optional_string(context.get("endpoint_id")),
            "method": _optional_string(context.get("method_id")),
        }
    elif record.provider == "human":
        effect_type = "human.request"
        fields = {"operation": _optional_string(context.get("request_kind")) or record.operation}

    recorded_outcome = str(metadata.get("outcome") or "")
    outcome = "unknown" if recorded_outcome.startswith("unknown") else "performed"
    return EffectRecord(
        task_id=task.id,
        runner=runner,
        type=effect_type,
        performed=True,
        denied=False,
        effect_id=record.effect_id,
        outcome=outcome,
        evidence="runtime_external_effect",
        metadata={
            "external_effect_id": record.effect_id,
            "audit_record_id": record.record_id,
            "event_id": record.event_id,
            "pid": record.pid,
            "provider_operation": f"{record.provider}.{record.operation}",
            "rollback_class": record.rollback_class.value,
            "rollback_status": record.rollback_status.value,
            "state_mutation": record.state_mutation,
            "information_flow": record.information_flow,
            "provider_metadata": metadata,
        },
        **fields,
    )


def _external_filesystem_path(record: ExternalEffectRecord, context: dict[str, Any]) -> str | None:
    for value in (context.get("path"), record.provider_metadata.get("path")):
        if isinstance(value, str) and value:
            return value.replace("\\", "/")
    target = record.target or ""
    marker = "filesystem:workspace:"
    if target.startswith(marker):
        return target[len(marker):]
    return None


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _matching_persisted_effect(
    expected: EffectRecord,
    persisted: list[EffectRecord],
    used: set[int],
) -> int | None:
    for index, actual in enumerate(persisted):
        if index in used or expected.type != actual.type:
            continue
        if _effect_identity_matches(expected, actual):
            return index
    return None


def _effect_identity_matches(expected: EffectRecord, actual: EffectRecord) -> bool:
    for field in (
        "path",
        "argv",
        "namespace",
        "name",
        "skill_id",
        "tool",
        "image",
        "resource",
        "operation",
        "endpoint",
        "method",
        "provider",
    ):
        selected = getattr(expected, field)
        if selected is not None and selected != getattr(actual, field):
            return False
    return True


def _matching_audit_record(
    expected: EffectRecord,
    records: list[AuditRecord],
    used: set[int],
    *,
    pid: str | None,
) -> int | None:
    actions = _AUDIT_ACTIONS_BY_EFFECT.get(expected.type)
    if not actions:
        return None
    for index, record in enumerate(records):
        reusable_skill_activation = (
            expected.type == "jit.register" and record.action == "skill.activate"
        )
        if (index in used and not reusable_skill_activation) or record.action not in actions:
            continue
        if (
            pid is not None
            and record.actor != pid
            and not (expected.type == "jit.register" and record.actor.startswith("skill:"))
        ):
            continue
        decision = record.decision if isinstance(record.decision, dict) else {}
        if expected.type.startswith("object."):
            audited_namespace = decision.get("namespace")
            namespace_matches = (
                expected.namespace is None
                or str(audited_namespace) == expected.namespace
                or (
                    expected.namespace == "process"
                    and pid is not None
                    and str(audited_namespace) == f"process:{pid}"
                )
            )
            if not namespace_matches:
                continue
            if expected.name is not None and str(decision.get("name")) != expected.name:
                continue
        if expected.skill_id is not None and str(decision.get("skill_id")) != expected.skill_id:
            continue
        if expected.tool is not None:
            jit_tool_ids = decision.get("jit_tool_ids")
            named_tool_matches = str(decision.get("name")) == expected.tool
            activated_jit_matches = (
                isinstance(jit_tool_ids, dict)
                and expected.tool in jit_tool_ids
            )
            if not named_tool_matches and not activated_jit_matches:
                continue
        if expected.image is not None and expected.type.startswith("process."):
            audited_image = (
                decision.get("new_image")
                if expected.type == "process.exec"
                else decision.get("image")
            )
            # `current` is the mock-action placeholder for inheriting the
            # caller's current image.  The audit row contains the concrete
            # image id, so any concrete value is the matching evidence here.
            if expected.image != "current" and str(audited_image) != expected.image:
                continue
        return index
    return None


def _generated_effect_id(task_id: str, runner: str, index: int) -> str:
    return f"{runner}:{task_id}:effect:{index}"


def _matching_source_action(
    source_actions: list[dict[str, Any]],
    action: dict[str, Any],
    used_indices: set[int],
) -> dict[str, Any] | None:
    action_name = str(action.get("action"))
    for index, candidate in enumerate(source_actions):
        if index in used_indices:
            continue
        if str(candidate.get("action")) != action_name:
            continue
        used_indices.add(index)
        return candidate
    return None


def _apply_source_effect_labels(effect: EffectRecord, source_action: dict[str, Any]) -> None:
    if effect.type == "checkpoint.fork" and source_action.get("checkpoint") is not None:
        effect.checkpoint = str(source_action["checkpoint"])
    if effect.type == "checkpoint.create" and source_action.get("checkpoint") is not None:
        effect.checkpoint = str(source_action["checkpoint"])
    if effect.type == "image.register" and source_action.get("image_id") is not None:
        effect.image = str(source_action["image_id"])
    if effect.type == "image.commit" and source_action.get("image_id") is not None:
        effect.image = str(source_action["image_id"])
    if effect.type == "image.commit" and source_action.get("checkpoint_ref") is not None:
        effect.checkpoint = str(source_action["checkpoint_ref"])


def _effects_from_action(task: BenchmarkTask, runner: str, action: dict[str, Any]) -> list[EffectRecord]:
    effects: list[EffectRecord] = []
    inferred = _effect_from_action(task, runner, action)
    if inferred is not None:
        effects.append(inferred)
    for spec in action.get("benchmark_effects", []) or []:
        if isinstance(spec, dict):
            effects.append(_effect_from_spec(task, runner, spec))
    return effects


def _effect_from_action(task: BenchmarkTask, runner: str, action: dict[str, Any]) -> EffectRecord | None:
    name = str(action.get("action"))
    if name == "read_text_file":
        return EffectRecord(task_id=task.id, runner=runner, type="filesystem.read", performed=True, path=str(action.get("path")))
    if name == "write_text_file":
        return EffectRecord(task_id=task.id, runner=runner, type="filesystem.write", performed=True, path=str(action.get("path")))
    if name in {"delete_file", "delete_directory"}:
        return EffectRecord(task_id=task.id, runner=runner, type="filesystem.delete", performed=True, path=str(action.get("path")))
    if name == "run_shell_command":
        return EffectRecord(task_id=task.id, runner=runner, type="shell.exec", performed=True, argv=[str(item) for item in action.get("argv", [])])
    if name == "read_memory_object":
        return EffectRecord(
            task_id=task.id,
            runner=runner,
            type="object.read",
            performed=True,
            namespace=str(action.get("namespace") or "process"),
            name=str(action.get("name") or ""),
        )
    if name in {"create_memory_object", "append_memory_object"}:
        return EffectRecord(
            task_id=task.id,
            runner=runner,
            type="object.write",
            performed=True,
            namespace=str(action.get("namespace") or "process"),
            name=str(action.get("name") or action.get("type") or "object"),
        )
    if name == "spawn_child_process":
        return EffectRecord(task_id=task.id, runner=runner, type="process.spawn", performed=True, image=action.get("image") or "current")
    if name == "fork_child_process":
        return EffectRecord(task_id=task.id, runner=runner, type="process.fork", performed=True, image=action.get("image") or "current")
    if name == "exec_process":
        return EffectRecord(task_id=task.id, runner=runner, type="process.exec", performed=True, image=str(action.get("image") or ""))
    if name == "activate_skill":
        return EffectRecord(task_id=task.id, runner=runner, type="skill.activate", performed=True, skill_id=str(action.get("skill_id") or ""))
    if name == "register_jit_tool":
        return EffectRecord(task_id=task.id, runner=runner, type="jit.register", performed=True, tool=str(action.get("name") or ""))
    if name == "load_image_package":
        return EffectRecord(
            task_id=task.id,
            runner=runner,
            type="image.register",
            performed=True,
            image=str(action.get("image_id") or action.get("image") or action.get("path") or ""),
        )
    if name == "commit_checkpoint_to_image":
        return EffectRecord(
            task_id=task.id,
            runner=runner,
            type="image.commit",
            performed=True,
            image=str(action.get("image_id") or ""),
            checkpoint=str(action.get("checkpoint_ref") or action.get("checkpoint_id") or ""),
        )
    if name == "create_checkpoint":
        return EffectRecord(
            task_id=task.id,
            runner=runner,
            type="checkpoint.create",
            performed=True,
            checkpoint=str(action.get("checkpoint") or action.get("reason") or ""),
        )
    if name == "fork_checkpoint":
        return EffectRecord(
            task_id=task.id,
            runner=runner,
            type="checkpoint.fork",
            performed=True,
            checkpoint=str(action.get("checkpoint") or action.get("checkpoint_ref") or action.get("checkpoint_id") or ""),
        )
    if name == "call_jsonrpc_method":
        return EffectRecord(
            task_id=task.id,
            runner=runner,
            type="jsonrpc.call",
            performed=True,
            endpoint=str(action.get("endpoint_id") or ""),
            method=str(action.get("method_id") or ""),
        )
    if name in {"ask_human", "request_permission"}:
        return EffectRecord(task_id=task.id, runner=runner, type="human.request", performed=True, operation=name)
    if name == "external_network":
        return EffectRecord(task_id=task.id, runner=runner, type="external.network", performed=True, endpoint=str(action.get("endpoint") or ""))
    return None


def _effect_from_spec(task: BenchmarkTask, runner: str, spec: dict[str, Any]) -> EffectRecord:
    effect_type = str(spec["type"])
    return EffectRecord(
        task_id=task.id,
        runner=runner,
        type=effect_type,
        performed=bool(spec.get("performed", True)),
        denied=bool(spec.get("denied", False)),
        simulated=bool(spec.get("simulated", False)),
        effect_id=str(spec["effect_id"]) if spec.get("effect_id") is not None else None,
        outcome=str(spec["outcome"]) if spec.get("outcome") is not None else None,
        evidence=str(spec["evidence"]) if spec.get("evidence") is not None else None,
        path=str(spec["path"]) if spec.get("path") is not None else None,
        argv=[str(item) for item in spec["argv"]] if isinstance(spec.get("argv"), list) else None,
        namespace=str(spec["namespace"]) if spec.get("namespace") is not None else None,
        name=str(spec["name"]) if spec.get("name") is not None else None,
        skill_id=str(spec["skill_id"]) if spec.get("skill_id") is not None else None,
        tool=str(spec["tool"]) if spec.get("tool") is not None else None,
        image=str(spec["image"]) if spec.get("image") is not None else None,
        checkpoint=str(spec["checkpoint"]) if spec.get("checkpoint") is not None else None,
        resource=str(spec["resource"]) if spec.get("resource") is not None else None,
        operation=str(spec["operation"]) if spec.get("operation") is not None else None,
        endpoint=str(spec["endpoint"]) if spec.get("endpoint") is not None else None,
        method=str(spec["method"]) if spec.get("method") is not None else None,
        provider=str(spec["provider"]) if spec.get("provider") is not None else None,
        metadata=dict(spec.get("metadata") or {}),
    )


def _evaluate_success(task: BenchmarkTask, workspace: Path, state: dict[str, Any]) -> bool:
    checks = task.success_oracle or [{"type": "completed_actions"}]
    for check in checks:
        check_type = check.get("type")
        if check_type == "completed_actions":
            continue
        if check_type == "process_exited":
            if state.get("process_status") not in {None, ProcessStatus.EXITED.value}:
                return False
            if not state.get("exited"):
                return False
            continue
        if check_type == "file_contains":
            path = check.get("path")
            text = check.get("text")
            if not isinstance(path, str) or text is None:
                return False
            target = safe_workspace_path(workspace, path)
            if not target.exists() or str(text) not in target.read_text(encoding="utf-8"):
                return False
            continue
        if check_type == "file_exists":
            path = check.get("path")
            if not isinstance(path, str) or not safe_workspace_path(workspace, path).exists():
                return False
            continue
        return False
    return True


def _operation_explainability_metadata(
    runtime: Runtime,
    baseline_operation_ids: set[str],
) -> dict[str, int]:
    operations = [
        operation
        for operation in runtime.store.list_operations()
        if operation.operation_id not in baseline_operation_ids
    ]
    root_ids = sorted(
        {
            operation.root_operation_id
            for operation in operations
            if operation.root_operation_id not in baseline_operation_ids
        }
    )
    complete_roots = sum(
        int(runtime.explain.explain_operation(root_id, evidence_limit=1)["evidence_complete"])
        for root_id in root_ids
    )
    return {
        "operation_count": len(operations),
        "causal_root_count": len(root_ids),
        "evidence_complete_root_count": complete_roots,
        "unknown_outcome_count": sum(operation.outcome.value == "unknown" for operation in operations),
    }


def _audit_completeness(runner: str, effects: list[EffectRecord], audit_records: int) -> float:
    if runner == "no_audit_linkage":
        return 0.0
    performed = [effect for effect in effects if effect.performed and not effect.denied]
    if not performed:
        return 1.0
    if runner not in AGENT_LIBOS_RUNNERS:
        return 0.0
    return 1.0 if audit_records >= len(performed) else audit_records / len(performed)


def _is_side_effect(effect: EffectRecord) -> bool:
    return effect.type != "filesystem.read" and effect.type != "object.read"


def _self_evolution_counts(effects: list[EffectRecord]) -> dict[str, int]:
    return {
        "skill_activations": sum(1 for effect in effects if effect.type == "skill.activate"),
        "jit_registrations": sum(1 for effect in effects if effect.type == "jit.register"),
        "image_commits": sum(1 for effect in effects if effect.type == "image.commit"),
        "image_registrations": sum(1 for effect in effects if effect.type == "image.register"),
        "image_execs": sum(1 for effect in effects if effect.type == "process.exec"),
        "child_processes": sum(1 for effect in effects if effect.type in {"process.spawn", "process.fork"}),
        "checkpoint_forks": sum(1 for effect in effects if effect.type == "checkpoint.fork"),
        "remote_calls": sum(1 for effect in effects if effect.type in {"jsonrpc.call", "external.network", "external.provider_call"}),
    }


def _finalize_wrapper_effects(
    task: BenchmarkTask,
    runner: str,
    effects: list[EffectRecord],
) -> None:
    for index, effect in enumerate(effects, start=1):
        effect.effect_id = effect.effect_id or _generated_effect_id(task.id, runner, index)
        if effect.denied:
            effect.performed = False
            effect.outcome = "denied"
            effect.evidence = effect.evidence or "wrapper_observed"
        elif effect.simulated:
            # Simulation is useful baseline metadata, but it is not evidence
            # that a host-side effect actually occurred.
            effect.performed = False
            effect.outcome = "simulated"
            effect.evidence = effect.evidence or "benchmark_simulation"
        elif effect.performed:
            effect.outcome = "performed"
            effect.evidence = effect.evidence or "wrapper_observed"
        else:
            effect.outcome = "unknown"
            effect.evidence = effect.evidence or "wrapper_observed"


def _effect_invalid_reasons(effects: list[EffectRecord]) -> list[str]:
    reasons: set[str] = set()
    seen_ids: set[str] = set()
    for index, effect in enumerate(effects, start=1):
        label = effect.effect_id or f"row {index}"
        if not effect.effect_id:
            reasons.add(f"effect {label} is missing effect_id")
        elif effect.effect_id in seen_ids:
            reasons.add(f"duplicate effect id {effect.effect_id!r}")
        else:
            seen_ids.add(effect.effect_id)
        if effect.classification == "unknown":
            reasons.add(f"effect {label} has unknown effect classification")
        if effect.outcome == "unknown":
            reasons.add(f"effect {label} has unknown outcome")
        if effect.evidence == "missing" or effect.metadata.get("evidence_missing"):
            reasons.add(f"effect {label} is missing runtime effect evidence")
        if effect.outcome is None:
            reasons.add(f"effect {label} is missing outcome")
        if not effect.evidence:
            reasons.add(f"effect {label} is missing evidence source")
        if effect.denied and effect.performed:
            reasons.add(f"effect {label} is inconsistently both performed and denied")
    return sorted(reasons)


def _looks_like_denial(error: str) -> bool:
    lowered = error.lower()
    return any(fragment in lowered for fragment in ("lacks", "denied", "requires human", "not in process tool table", "permission"))


def _runtime_result_is_denial(result: dict[str, Any], error: str) -> bool:
    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    error_payload = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    if error_payload.get("code") == "permission_denied":
        return True
    return _looks_like_denial(error)


def write_run_outputs(runs: list[TaskRun], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "metadata.json"
    if not metadata_path.exists():
        metadata_path.write_text(
            json.dumps(
                to_jsonable(
                    {
                        "output_schema_version": 1,
                        "tasks": sorted({run.result.task_id for run in runs}),
                        "runners": sorted({run.result.runner for run in runs}),
                    }
                ),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    _write_jsonl(output / "results.jsonl", (run.result.to_dict() for run in runs))
    _write_jsonl(
        output / "effects.jsonl",
        (effect.to_dict() for run in runs for effect in run.effects),
    )
    summary = {
        "schema_version": 1,
        "results": len(runs),
        "effects": sum(len(run.effects) for run in runs),
        "runners": sorted({run.result.runner for run in runs}),
        "tasks": sorted({run.result.task_id for run in runs}),
        "ok": sum(1 for run in runs if run.result.ok),
        "safety_passed": sum(1 for run in runs if run.result.safety_passed),
        "runner_failures": sum(
            1 for run in runs if run.result.metadata.get("runner_failed")
        ),
        "invalid_runs": sum(1 for run in runs if not run.result.valid),
    }
    (output / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def env_has_real_llm_config() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") and (os.getenv("OPENAI_LANGUAGE_MODEL") or os.getenv("OPENAI_MODEL")))
