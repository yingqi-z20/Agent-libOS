from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from contextlib import nullcontext
from dataclasses import replace
from typing import Any

from agent_libos.models import (
    AgentImage,
    DataFlowContext,
    DataLabels,
    FailedProcessOutcome,
    ObjectMetadata,
    OperationOutcome,
    ProcessExecutionToken,
    ProcessStatus,
    RuntimePublicationCursor,
)
from agent_libos.models.exceptions import (
    ProcessRevisionConflict,
    RuntimePublicationPending,
    RuntimeRecoveryRequired,
    ValidationError,
)
from agent_libos.process_execution import (
    bind_process_execution,
    current_process_execution_token,
)
from agent_libos.runtime.snapshots import ExecRollbackState, SnapshotCodec
from agent_libos.storage import OperationRepositoryProtocol, UnitOfWork
from agent_libos.utils.ids import new_id, utc_now


class ImageBootService:
    """Own image preflight, process initialization, exec, and compensation."""

    def __init__(
        self,
        *,
        process: Any,
        launch: Any,
        audit: Any,
        checkpoint: Any,
        authority_manifests: Any,
        modules: Any,
        tools: Any,
        skills: Any,
        exec_state: Any,
        checkpoint_installer: Any,
        package_installer: Any,
        unit_of_work: UnitOfWork,
        operations: Any,
        owner_instance_id: str,
        recovery_max_attempts: int,
        reconciliation_page_size: int,
        publication_lock: Any,
        recovery_required_callback: Any,
        require_recovery_lease: Any,
    ) -> None:
        self._process = process
        self._launch = launch
        self._audit = audit
        self._checkpoint = checkpoint
        self._authority_manifests = authority_manifests
        self._modules = modules
        self._tools = tools
        self._skills = skills
        self._exec_state = exec_state
        self._checkpoint_installer = checkpoint_installer
        self._package_installer = package_installer
        self._unit_of_work = unit_of_work
        self._processes = unit_of_work.processes
        self._publications = unit_of_work.publications
        self._operation_records: OperationRepositoryProtocol = (
            unit_of_work.evidence
        )
        self._authority = unit_of_work.authority
        self._objects = unit_of_work.objects
        self._extensions = unit_of_work.extensions
        self._operations = operations
        self._owner_instance_id = owner_instance_id
        self._recovery_max_attempts = int(recovery_max_attempts)
        self._reconciliation_page_size = int(reconciliation_page_size)
        self._publication_lock = publication_lock
        self._recovery_required_callback = recovery_required_callback
        self._require_recovery_lease = require_recovery_lease
        self._recovery_required_issuer_token = object()

    def exec(
        self,
        pid: str,
        image_id: str,
        *,
        execution_token: ProcessExecutionToken | None = None,
        args: dict[str, Any] | None = None,
        goal: dict[str, Any] | str | None = None,
        preserve_memory: bool = True,
        preserve_capabilities: bool = False,
        llm_profile_id: str | None = None,
        source_oids: Iterable[str] | None = None,
        source_labels: ObjectMetadata | DataLabels | dict[str, Any] | None = None,
        source_context: DataFlowContext | None = None,
    ) -> Any:
        # Snapshot-based exec compensation must not race process-local Tool or
        # Skill publication.  Those mutations use this same re-entrant lock, so
        # a legitimate concurrent proposal commits after the exec reaches a
        # terminal publication instead of being erased by its older snapshot.
        with self._publication_lock:
            return self._exec_locked(
                pid,
                image_id,
                execution_token=execution_token,
                args=args,
                goal=goal,
                preserve_memory=preserve_memory,
                preserve_capabilities=preserve_capabilities,
                llm_profile_id=llm_profile_id,
                source_oids=source_oids,
                source_labels=source_labels,
                source_context=source_context,
            )

    def _exec_locked(
        self,
        pid: str,
        image_id: str,
        *,
        execution_token: ProcessExecutionToken | None,
        args: dict[str, Any] | None = None,
        goal: dict[str, Any] | str | None = None,
        preserve_memory: bool = True,
        preserve_capabilities: bool = False,
        llm_profile_id: str | None = None,
        source_oids: Iterable[str] | None = None,
        source_labels: ObjectMetadata | DataLabels | dict[str, Any] | None = None,
        source_context: DataFlowContext | None = None,
    ) -> Any:
        process = self._process.get(pid)
        self._assert_exec_process_state(
            process,
            execution_token=execution_token,
        )
        image = self._launch.require_image(image_id)
        if image_id != process.image_id:
            self._launch.require_image_boot_authority(pid, image_id)
        self.preflight(image)
        publication_id = new_id("publication")
        previous_state: ExecRollbackState | None = None
        admission_token: ProcessExecutionToken | None = None
        try:
            previous_state, admission_token, publication = (
                self._capture_admit_and_begin_exec(
                    process,
                    image,
                    publication_id=publication_id,
                    execution_token=execution_token,
                )
            )
            if admission_token is None or publication is None:
                raise ProcessRevisionConflict(
                    f"process exec admission conflict for {pid}"
                )
            admitted_publication_id, boot_kind, workspace_root = publication
            if admitted_publication_id != publication_id:
                raise ValidationError(
                    f"process exec admission returned another publication: {pid}"
                )
            previous_state = replace(
                previous_state,
                capability_rollback_token=publication_id,
            )
            self._apply_admitted_exec(
                publication_id=publication_id,
                pid=pid,
                image=image,
                previous_process=process,
                admission_token=admission_token,
                prior_execution_token=execution_token,
                boot_kind=boot_kind,
                workspace_root=workspace_root,
                args=args,
                goal=goal,
                preserve_memory=preserve_memory,
                preserve_capabilities=preserve_capabilities,
                llm_profile_id=llm_profile_id,
                source_oids=source_oids,
                source_labels=source_labels,
                source_context=source_context,
            )
        except BaseException as exc:
            self._resolve_failed_exec(
                publication_id=publication_id,
                pid=pid,
                image_id=image_id,
                previous_state=previous_state,
                admission_token=admission_token,
                error=exc,
            )
            raise
        return self._process.get(pid)

    def _apply_admitted_exec(
        self,
        *,
        publication_id: str,
        pid: str,
        image: AgentImage,
        previous_process: Any,
        admission_token: ProcessExecutionToken,
        prior_execution_token: ProcessExecutionToken | None,
        boot_kind: str,
        workspace_root: str | None,
        args: dict[str, Any] | None,
        goal: dict[str, Any] | str | None,
        preserve_memory: bool,
        preserve_capabilities: bool,
        llm_profile_id: str | None,
        source_oids: Iterable[str] | None,
        source_labels: ObjectMetadata | DataLabels | dict[str, Any] | None,
        source_context: DataFlowContext | None,
    ) -> None:
        with bind_process_execution(admission_token):
            self._process.apply_exec_state(
                pid=pid,
                image=image.image_id,
                args=args,
                goal=goal,
                preserve_memory=preserve_memory,
                preserve_capabilities=preserve_capabilities,
                llm_profile_id=llm_profile_id,
                source_oids=source_oids,
                source_labels=source_labels,
                source_context=source_context,
                _record_evidence=False,
                _capability_rollback_token=publication_id,
            )
            self._advance_publication(publication_id, "process_exec_applied")
            assigned_by = f"publication:{publication_id}"
            initial_tool_table, initial_model_tool_table = self._configure_tools(
                pid,
                image,
                assigned_by,
            )
            self._advance_publication(publication_id, "tools_configured")
            self._instantiate_boot(
                pid,
                image,
                boot_kind,
                workspace_root=workspace_root,
                publication_id=publication_id,
            )
            self._advance_publication(publication_id, "boot_instantiated")
            self._configure_skills(
                pid,
                image,
                assigned_by,
                publication_id=publication_id,
                initial_tool_table=initial_tool_table,
                initial_model_tool_table=initial_model_tool_table,
            )
            self._advance_publication(publication_id, "skills_configured")
            self._commit_exec_publication(
                publication_id,
                pid,
                previous_process,
                prior_execution_token=prior_execution_token,
                args=args,
                preserve_memory=preserve_memory,
                preserve_capabilities=preserve_capabilities,
                goal_changed=goal is not None,
            )

    def _resolve_failed_exec(
        self,
        *,
        publication_id: str,
        pid: str,
        image_id: str,
        previous_state: ExecRollbackState | None,
        admission_token: ProcessExecutionToken | None,
        error: BaseException,
    ) -> None:
        try:
            publication = self._publications.get_runtime_publication(publication_id)
            if publication is None:
                return
            if self._confirmed_exec_terminal_outcome(publication, pid=pid) is not None:
                return
            rollback_state = previous_state or self._exec_rollback_state(publication)
            rollback_token = admission_token or self._exec_admission_token(publication)
        except BaseException as diagnostic_error:
            self._fence_exec_outcome_uncertainty(
                publication_id,
                error,
                diagnostic_error,
            )
        try:
            execution_scope = (
                bind_process_execution(rollback_token)
                if rollback_token is not None
                else nullcontext()
            )
            with execution_scope:
                self._rollback_failed_exec(
                    publication_id=publication_id,
                    pid=pid,
                    image_id=image_id,
                    previous_state=rollback_state,
                    fence_execution=True,
                    error=error,
                )
        except BaseException as rollback_error:
            try:
                publication = self._publications.get_runtime_publication(publication_id)
                if (
                    publication is not None
                    and self._confirmed_exec_terminal_outcome(publication, pid=pid)
                    is not None
                ):
                    return
            except BaseException as diagnostic_error:
                self._fence_exec_outcome_uncertainty(
                    publication_id,
                    error,
                    rollback_error,
                    diagnostic_error,
                )
            self._raise_exec_rollback_failure(
                publication_id=publication_id,
                pid=pid,
                error=error,
                rollback_error=rollback_error,
            )

    def _fence_exec_outcome_uncertainty(
        self,
        publication_id: str,
        error: BaseException,
        *secondary: BaseException,
    ) -> None:
        failure = BaseExceptionGroup(
            "process exec durable outcome is uncertain",
            [error, *secondary],
        )
        try:
            self._recovery_required_callback(publication_id=publication_id)
        except BaseException as fence_error:
            raise BaseExceptionGroup(
                "process exec durable outcome fence failed",
                [failure, fence_error],
            ) from error
        raise failure from error

    def _confirmed_exec_terminal_outcome(
        self,
        publication: dict[str, Any],
        *,
        pid: str,
    ) -> str | None:
        state = str(publication.get("state") or "")
        terminal = {
            "committed": ("committed", OperationOutcome.SUCCEEDED),
            "rolled_back": ("compensated", OperationOutcome.FAILED),
        }
        expected = terminal.get(state)
        if expected is None:
            return None
        phase, outcome = expected
        publication_id = str(publication.get("publication_id") or "")
        plan = dict(publication.get("plan") or {})
        if (
            not publication_id
            or publication.get("kind") != "process_exec"
            or str(publication.get("pid") or "") != pid
            or str(plan.get("pid") or "") != pid
            or publication.get("phase") != phase
            or publication.get("operation_reconciled") is not True
            or publication.get("owner_instance_id") != self._owner_instance_id
        ):
            raise ValidationError(
                f"process exec terminal publication identity is invalid: {publication_id}"
            )
        receipt = self._exec_terminal_receipt(
            publication,
            expected_phase=phase,
            pid=pid,
        )
        self._assert_exec_terminal_process(
            publication,
            receipt=receipt,
            state=state,
            pid=pid,
        )
        self._assert_exec_terminal_operation(
            publication,
            expected_outcome=outcome,
            expected_phase=phase,
        )
        return state

    @staticmethod
    def _exec_terminal_receipt(
        publication: dict[str, Any],
        *,
        expected_phase: str,
        pid: str,
    ) -> dict[str, Any]:
        phases = list(dict(publication.get("receipt") or {}).get("phases") or [])
        terminal = [
            dict(item)
            for item in phases
            if isinstance(item, dict) and item.get("phase") == expected_phase
        ]
        if len(terminal) != 1 or str(terminal[0].get("pid") or "") != pid:
            raise ValidationError(
                "process exec terminal publication has no exact terminal receipt: "
                f"{publication.get('publication_id')}"
            )
        for field in ("revision", "execution_generation", "state_generation"):
            value = terminal[0].get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValidationError(
                    f"process exec terminal receipt has invalid {field}: "
                    f"{publication.get('publication_id')}"
                )
        return terminal[0]

    def _assert_exec_terminal_process(
        self,
        publication: dict[str, Any],
        *,
        receipt: dict[str, Any],
        state: str,
        pid: str,
    ) -> None:
        expected_image = str(publication["plan"].get("image_id") or "")
        if state == "rolled_back":
            rollback = self._exec_rollback_state(publication)
            before_rows = rollback.snapshot.rows.processes
            if len(before_rows) != 1:
                raise ValidationError(
                    f"process exec rollback snapshot is not exact: {publication['publication_id']}"
                )
            expected_image = str(before_rows[0].get("image_id") or "")
        if (
            not expected_image
            or receipt.get("image_id") != expected_image
            or receipt.get("status") != ProcessStatus.RUNNABLE.value
        ):
            raise ValidationError(
                "process exec terminal receipt has invalid process state: "
                f"{publication['publication_id']}"
            )
        process = self._processes.get_process(pid)
        if process is None:
            raise ValidationError(f"terminal process exec lost its process: {pid}")
        if (
            process.revision < receipt["revision"]
            or process.execution_generation < receipt["execution_generation"]
            or process.state_generation < receipt["state_generation"]
        ):
            raise ValidationError(
                "process concurrency regressed behind the terminal exec receipt: "
                f"{publication['publication_id']}"
            )
        if process.revision != receipt["revision"]:
            return
        if (
            process.image_id != expected_image
            or process.status != ProcessStatus.RUNNABLE
            or process.execution_generation != receipt["execution_generation"]
            or process.state_generation != receipt["state_generation"]
            or process.execution_owner_id is not None
            or process.execution_lease_id is not None
        ):
            raise ValidationError(
                "process state does not match the terminal exec receipt: "
                f"{publication['publication_id']}"
            )

    def _assert_exec_terminal_operation(
        self,
        publication: dict[str, Any],
        *,
        expected_outcome: OperationOutcome,
        expected_phase: str,
    ) -> None:
        publication_id = str(publication["publication_id"])
        pid = str(publication["pid"])
        operation_id = str(publication["plan"].get("operation_id") or "")
        if not operation_id:
            reverse = self._operations.runtime_publication_binding_operation_ids(
                publication_id
            )
            if publication["plan"].get("operation_binding_version") is not None or reverse:
                raise ValidationError(
                    f"terminal process exec has invalid unbound operation: {publication_id}"
                )
            return
        operation = self._operation_records.get_operation(operation_id)
        metadata = operation.metadata if operation is not None else {}
        if (
            operation is None
            or operation.kind.value != "runtime"
            or operation.name != "process.exec"
            or operation.actor != pid
            or operation.pid != pid
            or operation.state.value != "terminal"
            or operation.outcome != expected_outcome
            or metadata.get("runtime_publication_id") != publication_id
            or metadata.get("runtime_publication_kind") != "process_exec"
            or metadata.get("runtime_publication_bound") is not True
            or metadata.get("runtime_publication_state") != publication["state"]
            or metadata.get("runtime_publication_phase") != expected_phase
        ):
            raise ValidationError(
                f"terminal process exec operation binding is invalid: {publication_id}"
            )

    def _raise_exec_rollback_failure(
        self,
        *,
        publication_id: str,
        pid: str,
        error: BaseException,
        rollback_error: BaseException,
    ) -> None:
        if isinstance(error, Exception) and isinstance(rollback_error, Exception):
            try:
                recovery_required = self._recovery_required_signal(publication_id)
            except BaseException as diagnostic_error:
                failure = BaseExceptionGroup(
                    "process exec rollback diagnosis failed",
                    [error, rollback_error, diagnostic_error],
                )
                try:
                    self._recovery_required_callback(publication_id=publication_id)
                except BaseException as fence_error:
                    raise BaseExceptionGroup(
                        "process exec recovery fence failed",
                        [failure, fence_error],
                    ) from error
                raise failure from error
            if recovery_required is not None:
                try:
                    self.fence_recovery_required_signal(
                        recovery_required,
                        pid=pid,
                    )
                except Exception as validation_error:
                    raise validation_error from recovery_required
                except BaseException as fence_error:
                    raise BaseExceptionGroup(
                        "process exec recovery fence failed",
                        [error, rollback_error, fence_error],
                    ) from error
                raise recovery_required from rollback_error
            raise rollback_error

        failure: BaseException = BaseExceptionGroup(
            "process exec and rollback failed",
            [error, rollback_error],
        )
        try:
            recovery_required = self._recovery_required_signal(publication_id)
        except BaseException as diagnostic_error:
            failure = BaseExceptionGroup(
                "process exec rollback diagnosis failed",
                [failure, diagnostic_error],
            )
            try:
                self._recovery_required_callback(publication_id=publication_id)
            except BaseException as fence_error:
                raise BaseExceptionGroup(
                    "process exec recovery fence failed",
                    [failure, fence_error],
                ) from error
            raise failure from error
        if recovery_required is not None:
            try:
                self.fence_recovery_required_signal(
                    recovery_required,
                    pid=pid,
                )
            except BaseException as fence_error:
                raise BaseExceptionGroup(
                    "process exec recovery fence failed",
                    [failure, fence_error],
                ) from error
            raise BaseExceptionGroup(
                "process exec interruption requires recovery",
                [failure, recovery_required],
            ) from error
        try:
            current_publication = self._publications.get_runtime_publication(
                publication_id
            )
            unresolved = current_publication is None or (
                current_publication["state"] not in {"committed", "rolled_back"}
            )
            if unresolved:
                self._recovery_required_callback(publication_id=publication_id)
        except BaseException as fence_error:
            raise BaseExceptionGroup(
                "process exec terminalization fence failed",
                [failure, fence_error],
            ) from error
        raise failure from error

    def _capture_admit_and_begin_exec(
        self,
        process: Any,
        image: AgentImage,
        *,
        publication_id: str,
        execution_token: ProcessExecutionToken | None,
    ) -> tuple[
        ExecRollbackState,
        ProcessExecutionToken | None,
        tuple[str, str, str | None] | None,
    ]:
        """Capture rollback state and acquire exec ownership atomically.

        Host and worker paths both rotate the complete process concurrency
        tuple before publication.  A scheduler completion or claim that wins
        first prevents publication, while the winning exec owns an internal
        token that snapshot compensation always fences.
        """

        admission_token: ProcessExecutionToken | None = None
        publication: tuple[str, str, str | None] | None = None
        with self._unit_of_work.transaction(include_object_payloads=True):
            previous_state = self._exec_state.capture(process.pid)
            if execution_token is None:
                admission_token = self._processes.claim_host_process_exec(
                    process.pid,
                    owner_id=f"{self._owner_instance_id}:process.exec",
                    expected_revision=process.revision,
                    expected_state_generation=process.state_generation,
                    expected_execution_generation=process.execution_generation,
                )
            else:
                admission_token = self._processes.claim_worker_process_exec(
                    process.pid,
                    execution_token=execution_token,
                    owner_id=f"{self._owner_instance_id}:process.exec",
                    expected_revision=process.revision,
                    expected_state_generation=process.state_generation,
                )
            if admission_token is not None:
                with bind_process_execution(admission_token):
                    publication = self._begin_exec_publication(
                        process.pid,
                        image,
                        previous_state,
                        publication_id=publication_id,
                    )
        return previous_state, admission_token, publication

    @staticmethod
    def _assert_exec_process_state(
        process: Any,
        *,
        execution_token: ProcessExecutionToken | None,
    ) -> None:
        """Reject waits before exec can publish or mutate replacement state.

        A typed wait owns durable dependency state (a child, mailbox filter,
        Human request, Tool operation, or Host resume gate).  Exec has no
        all-or-nothing supersede operation for those dependencies, so silently
        clearing the wait would orphan them.  Only a Host-owned RUNNABLE row or
        the exact currently claimed RUNNING generation is safe to replace.
        """

        if process.wait_state is not None:
            raise ValidationError(
                "process exec requires resolving the active typed wait before "
                f"image replacement: {process.pid} ({process.wait_state.kind})"
            )
        token = current_process_execution_token()
        if execution_token != token:
            raise ValidationError(
                "process exec execution token does not match the ambient "
                f"execution lease: {process.pid}"
            )
        if process.status == ProcessStatus.RUNNABLE and token is None:
            return
        if (
            process.status == ProcessStatus.RUNNING
            and token is not None
            and token.pid == process.pid
            and token.generation == process.execution_generation
            and token.owner_id == process.execution_owner_id
            and token.lease_id == process.execution_lease_id
        ):
            return
        raise ValidationError(
            "process exec requires a runnable process or its exact active "
            f"execution lease: {process.pid} ({process.status.value})"
        )

    def fence_recovery_required_signal(
        self,
        error: RuntimeRecoveryRequired,
        *,
        pid: str,
    ) -> bool:
        """Fence mutations, then validate an internally issued durable signal."""

        if error._issuer_token is not self._recovery_required_issuer_token:
            return False
        publication = self._publications.get_runtime_publication(error.publication_id)
        if publication is None:
            raise ValidationError(
                "runtime recovery signal references a missing publication: "
                f"{error.publication_id}"
            )
        if not self._publication_requires_snapshot_replay(publication):
            return False
        # Once durable state says snapshot replay is still required, fail all
        # mutation admission closed before reporting any association damage.
        self._recovery_required_callback(publication_id=error.publication_id)
        operation_id = str(publication["plan"].get("operation_id") or "")
        identity = {
            "kind": str(publication["kind"]),
            "pid": str(publication["pid"]),
            "operation_id": operation_id,
            "state": str(publication["state"]),
            "phase": str(publication["phase"]),
        }
        expected = {
            "kind": "process_exec",
            "pid": str(pid),
            "operation_id": str(error.operation_id),
            "state": str(error.state),
            "phase": str(error.phase),
        }
        if identity != expected or str(error.pid) != str(pid):
            raise ValidationError(
                "runtime recovery signal does not match its durable publication: "
                f"{error.publication_id}"
            )
        operation = self._operation_records.get_operation(operation_id)
        if (
            operation is None
            or operation.kind.value != "runtime"
            or operation.name != "process.exec"
            or operation.actor != str(pid)
            or operation.pid != str(pid)
            or operation.metadata.get("runtime_publication_id")
            != error.publication_id
            or operation.metadata.get("runtime_publication_kind")
            != "process_exec"
            or operation.metadata.get("runtime_publication_bound") is not True
        ):
            raise ValidationError(
                "runtime recovery signal operation binding is invalid: "
                f"{error.publication_id} -> {operation_id or '<missing>'}"
            )
        return True

    def _recovery_required_signal(
        self,
        publication_id: str,
    ) -> RuntimeRecoveryRequired | None:
        publication = self._publications.get_runtime_publication(publication_id)
        if publication is None or not self._publication_requires_snapshot_replay(
            publication
        ):
            return None
        operation_id = str(publication["plan"].get("operation_id") or "")
        return RuntimeRecoveryRequired(
            publication_id=publication_id,
            operation_id=operation_id,
            pid=str(publication["pid"]),
            state=str(publication["state"]),
            phase=str(publication["phase"]),
            _issuer_token=self._recovery_required_issuer_token,
        )

    def _publication_requires_snapshot_replay(
        self,
        publication: dict[str, Any],
    ) -> bool:
        if publication["kind"] != "process_exec" or publication["state"] not in {
            "planning",
            "applying",
            "rollback_pending",
            "failed",
            "manual",
        }:
            return False
        try:
            return not self._has_compensation_applied_marker(publication)
        except ValidationError:
            return True

    def _begin_exec_publication(
        self,
        pid: str,
        image: AgentImage,
        previous_state: ExecRollbackState,
        *,
        publication_id: str,
    ) -> tuple[str, str, str | None]:
        admission_token = current_process_execution_token()
        if admission_token is None or admission_token.pid != pid:
            raise ValidationError(
                f"process exec publication requires its exact admission token: {pid}"
            )
        boot_kind = str(image.boot.get("kind", "fresh"))
        workspace_root = (
            self._package_installer.planned_workspace_root(
                pid,
                image,
                materialization_id=publication_id,
            )
            if boot_kind == "image_package"
            else None
        )
        operation_id = self._operations.current_id()
        with self._unit_of_work.transaction():
            self._publications.insert_runtime_publication(
                publication_id=publication_id,
                kind="process_exec",
                pid=pid,
                owner_instance_id=self._owner_instance_id,
                plan={
                    "pid": pid,
                    "image_id": image.image_id,
                    "artifact_owner": f"publication:{publication_id}",
                    "before_snapshot": previous_state.snapshot.to_mapping(),
                    "before_tool_ids": sorted(previous_state.tool_ids),
                    "boot_kind": boot_kind,
                    "materialized_workspace_root": workspace_root,
                    "operation_id": operation_id,
                    "operation_binding_version": 1 if operation_id is not None else None,
                    "admission_execution_generation": admission_token.generation,
                    "admission_execution_owner_id": admission_token.owner_id,
                    "admission_execution_lease_id": admission_token.lease_id,
                },
            )
            if operation_id is not None:
                self._operations.bind_runtime_publication(
                    operation_id,
                    publication_id=publication_id,
                    publication_kind="process_exec",
                    expected_kind="runtime",
                    expected_name="process.exec",
                    expected_actor=pid,
                    expected_pid=pid,
                )
        return publication_id, boot_kind, workspace_root

    def _commit_exec_publication(
        self,
        publication_id: str,
        pid: str,
        previous_process: Any,
        *,
        prior_execution_token: ProcessExecutionToken | None,
        args: dict[str, Any] | None,
        preserve_memory: bool,
        preserve_capabilities: bool,
        goal_changed: bool,
    ) -> None:
        with self._unit_of_work.transaction():
            self._process.finalize_exec_capability_revocations(
                pid,
                publication_id,
            )
            current = self._process.get(pid)
            current = self._processes.commit_process_exec_epoch(
                pid,
                publication_id=publication_id,
                expected_revision=current.revision,
            )
            event, audit = self._process.record_exec_evidence(
                pid,
                old_image=previous_process.image_id,
                args=args,
                preserve_memory=preserve_memory,
                preserve_capabilities=preserve_capabilities,
                new_goal_oid=current.goal_oid if goal_changed else None,
            )
            if not self._publications.advance_runtime_publication(
                publication_id,
                state="committed",
                phase="committed",
                receipt={
                    "phase": "committed",
                    "pid": pid,
                    "revision": current.revision,
                    "execution_generation": current.execution_generation,
                    "state_generation": current.state_generation,
                    "image_id": current.image_id,
                    "status": current.status.value,
                    "prior_execution_generation": (
                        prior_execution_token.generation
                        if prior_execution_token is not None
                        else None
                    ),
                    "prior_execution_owner_id": (
                        prior_execution_token.owner_id
                        if prior_execution_token is not None
                        else None
                    ),
                    "prior_execution_lease_id": (
                        prior_execution_token.lease_id
                        if prior_execution_token is not None
                        else None
                    ),
                    "event_id": event.event_id,
                    "audit_id": audit.record_id,
                },
                expected_states={"applying"},
            ):
                raise ValidationError(
                    f"cannot commit process exec publication: {publication_id}"
                )
            self._reconcile_publication_operation(
                publication_id,
                OperationOutcome.SUCCEEDED,
                publication_state="committed",
                publication_phase="committed",
            )

    def _compensate_failed_exec(
        self,
        publication_id: str,
        pid: str,
        previous_state: ExecRollbackState,
        *,
        fence_execution: bool,
    ) -> list[BaseException]:
        cleanup_errors: list[BaseException] = []
        try:
            # The durable marker and snapshot restore are one atomic boundary.
            # If the later publication/operation terminal transaction fails,
            # startup can finish terminalization without replaying this stale
            # snapshot over legitimate mutations admitted after exec returned.
            with self._unit_of_work.transaction(include_object_payloads=True):
                publication = self._publications.get_runtime_publication(publication_id)
                if publication is None:
                    raise ValidationError(
                        f"process exec publication disappeared: {publication_id}"
                    )
                self._cleanup_publication_artifacts(
                    publication,
                    reason="process_exec_publication_compensation",
                )
                self._exec_state.restore(
                    previous_state,
                    fence_execution=fence_execution,
                )
                publication = self._publications.get_runtime_publication(publication_id)
                if publication is None:
                    raise ValidationError(
                        f"process exec publication disappeared: {publication_id}"
                    )
                self.assert_publication_artifacts_removed(publication)
                if not self._publications.advance_runtime_publication(
                    publication_id,
                    state="rollback_pending",
                    phase="compensation_applied",
                    receipt={"phase": "compensation_applied", "pid": pid},
                    expected_states={"rollback_pending"},
                ):
                    raise ValidationError(
                        "cannot persist applied process exec compensation: "
                        f"{publication_id}"
                    )
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        return cleanup_errors

    def _rollback_failed_exec(
        self,
        *,
        publication_id: str,
        pid: str,
        image_id: str,
        previous_state: ExecRollbackState,
        fence_execution: bool,
        error: BaseException,
    ) -> None:
        if not self._publications.advance_runtime_publication(
            publication_id,
            state="rollback_pending",
            phase="compensating",
            error={"code": "process_exec_failed", "error_type": type(error).__name__},
            expected_states={"planning", "applying"},
        ):
            raise ValidationError(
                f"cannot begin process exec compensation: {publication_id}"
            ) from error
        cleanup_errors = self._compensate_failed_exec(
            publication_id,
            pid,
            previous_state,
            fence_execution=fence_execution,
        )
        if cleanup_errors:
            try:
                with self._unit_of_work.transaction():
                    if not self._publications.advance_runtime_publication(
                        publication_id,
                        state="failed",
                        phase="compensation_failed",
                        error={
                            "code": "process_exec_compensation_failed",
                            "error_type": type(cleanup_errors[-1]).__name__,
                        },
                        expected_states={"rollback_pending"},
                    ):
                        raise ValidationError(
                            f"cannot record failed process exec compensation: {publication_id}"
                        )
                    self._reconcile_publication_operation(
                        publication_id,
                        OperationOutcome.UNKNOWN,
                        publication_state="failed",
                        publication_phase="compensation_failed",
                    )
            except BaseException as terminal_error:
                try:
                    self._raise_pending_publication_outcome(
                        publication_id,
                        terminal_error,
                    )
                except BaseException as pending_error:
                    if not isinstance(error, Exception) or any(
                        not isinstance(item, Exception)
                        for item in cleanup_errors
                    ):
                        secondary = [*cleanup_errors, terminal_error]
                        if pending_error is not terminal_error:
                            secondary.append(pending_error)
                        raise BaseExceptionGroup(
                            "process exec compensation terminalization failed",
                            secondary,
                        ) from error
                    raise
            raise BaseExceptionGroup(
                "process exec compensation failed",
                cleanup_errors,
            ) from error
        self._finalize_exec_rollback(
            publication_id=publication_id,
            pid=pid,
            image_id=image_id,
            error=error,
        )

    def _finalize_exec_rollback(
        self,
        *,
        publication_id: str,
        pid: str,
        image_id: str,
        error: BaseException,
    ) -> None:
        try:
            with self._unit_of_work.transaction():
                restored = self._processes.get_process(pid)
                if restored is None:
                    raise ValidationError(
                        f"process disappeared after exec compensation: {pid}"
                    )
                if not self._publications.advance_runtime_publication(
                    publication_id,
                    state="rolled_back",
                    phase="compensated",
                    receipt={
                        "phase": "compensated",
                        "pid": pid,
                        "revision": restored.revision,
                        "execution_generation": restored.execution_generation,
                        "state_generation": restored.state_generation,
                        "image_id": restored.image_id,
                        "status": restored.status.value,
                    },
                    expected_states={"rollback_pending"},
                ):
                    raise ValidationError(
                        f"cannot publish process exec compensation: {publication_id}"
                    )
                self._reconcile_publication_operation(
                    publication_id,
                    OperationOutcome.FAILED,
                    publication_state="rolled_back",
                    publication_phase="compensated",
                )
                self._audit.record(
                    actor="runtime",
                    action="image.boot.failed",
                    target=f"process:{pid}",
                    decision={
                        "image": image_id,
                        "phase": "process.exec",
                        "error": str(error),
                        "rolled_back": True,
                    },
                )
        except BaseException as terminal_error:
            try:
                self._raise_pending_publication_outcome(
                    publication_id,
                    terminal_error,
                )
            except BaseException as pending_error:
                if not isinstance(error, Exception):
                    secondary = [terminal_error]
                    if pending_error is not terminal_error:
                        secondary.append(pending_error)
                    raise BaseExceptionGroup(
                        "process exec terminalization failed",
                        secondary,
                    ) from error
                raise

    def _raise_pending_publication_outcome(
        self,
        publication_id: str,
        cause: BaseException,
    ) -> None:
        """Keep the linked operation open when its terminal receipt rolled back."""

        publication = self._publications.get_runtime_publication(publication_id)
        operation_id = (
            str(publication["plan"].get("operation_id") or "")
            if publication is not None
            else ""
        )
        current_operation_id = str(self._operations.current_id() or "")
        operation = (
            self._operation_records.get_operation(operation_id)
            if operation_id and operation_id == current_operation_id
            else None
        )
        if (
            publication is not None
            and publication["state"] in {"planning", "applying", "rollback_pending"}
            and operation is not None
            and operation.state.value != "terminal"
        ):
            raise RuntimePublicationPending(
                publication_id=publication_id,
                operation_id=operation_id,
                state=publication["state"],
                phase=publication["phase"],
            ) from cause
        raise cause

    def recover_incomplete_publications(self) -> list[str]:
        self._require_recovery_lease()
        recovered: list[str] = []
        for state in ("planning", "applying", "rollback_pending", "failed", "manual"):
            for operation_reconciled in (False, True):
                self._recover_exec_publication_state(
                    state,
                    operation_reconciled=operation_reconciled,
                    recovered=recovered,
                )
        self.reconcile_terminal_publications()
        return recovered

    def _recover_exec_publication_state(
        self,
        state: str,
        *,
        operation_reconciled: bool,
        recovered: list[str],
    ) -> None:
        after: RuntimePublicationCursor | None = None
        while True:
            page = self._publications.query_runtime_publication_recovery(
                kind="process_exec",
                state=state,
                operation_reconciled=operation_reconciled,
                after=after,
                limit=self._reconciliation_page_size,
            )
            previous = after
            for publication in page.records:
                cursor = RuntimePublicationCursor(
                    publication["created_at"],
                    publication["publication_id"],
                )
                if (
                    publication["kind"] != "process_exec"
                    or publication["state"] != state
                    or publication["operation_reconciled"] is not operation_reconciled
                    or (previous is not None and cursor <= previous)
                ):
                    raise ValidationError(
                        "runtime publication repository returned an invalid exec recovery page"
                    )
                recovered_id = self._recover_exec_publication(publication)
                if (
                    recovered_id is not None
                    and len(recovered) < self._reconciliation_page_size
                ):
                    recovered.append(recovered_id)
                previous = cursor
            if page.next_cursor is None:
                break
            if previous is None or page.next_cursor != previous:
                raise ValidationError(
                    "runtime publication repository returned an invalid exec recovery cursor"
                )
            after = page.next_cursor

    def _recover_exec_publication(
        self,
        publication: dict[str, Any],
    ) -> str | None:
        if publication["state"] == "manual":
            self._fail_closed_manual_exec_publication(publication)
        claimed = self._publications.claim_runtime_publication_recovery(
            publication["publication_id"],
            claimant_instance_id=self._owner_instance_id,
            expected_owner_instance_id=publication["owner_instance_id"],
            expected_state=publication["state"],
            classification="compensate_process_exec",
            max_attempts=self._recovery_max_attempts,
            allow_orphaned_claim_takeover=True,
        )
        if claimed is None:
            self._require_exec_publication_resolved(publication["publication_id"])
            return None
        if claimed["state"] == "manual":
            self._fail_closed_manual_exec_publication(claimed)
        recovery_lease_id = self._recovery_lease_id(claimed)
        try:
            self._compensate_claimed_exec_publication(
                claimed,
                recovery_lease_id=recovery_lease_id,
            )
        except Exception as exc:
            self._record_exec_recovery_failure(
                claimed,
                recovery_lease_id=recovery_lease_id,
                error=exc,
            )
            raise ValidationError(
                f"cannot recover process exec publication: {claimed['publication_id']}"
            ) from exc
        return str(claimed["publication_id"])

    def _compensate_claimed_exec_publication(
        self,
        claimed: dict[str, Any],
        *,
        recovery_lease_id: str,
    ) -> None:
        publication_id = str(claimed["publication_id"])
        with self._unit_of_work.transaction(include_object_payloads=True):
            current = self._publications.get_runtime_publication(publication_id)
            if current is None:
                raise ValidationError(
                    f"process exec publication disappeared: {publication_id}"
                )
            if self._recovery_lease_id(current) != recovery_lease_id:
                raise ValidationError(
                    f"process exec recovery lease changed: {publication_id}"
                )
            compensation_applied = self._has_compensation_applied_marker(current)
            if not compensation_applied:
                state = self._exec_rollback_state(current)
                with bind_process_execution(self._exec_admission_token(current)):
                    self._cleanup_publication_artifacts(
                        current,
                        reason="process_exec_startup_compensation",
                    )
                    self._exec_state.restore(state)
            self.assert_publication_artifacts_removed(current)
            terminal_phase = (
                "startup_compensation_finalized"
                if compensation_applied
                else "startup_compensated"
            )
            if not self._publications.advance_runtime_publication(
                publication_id,
                state="rolled_back",
                phase=terminal_phase,
                receipt={"phase": terminal_phase, "pid": current["pid"]},
                expected_states={"rollback_pending"},
                recovery_lease_id=recovery_lease_id,
            ):
                raise ValidationError(
                    "process exec recovery lease changed before terminal publication: "
                    f"{publication_id}"
                )
            self._reconcile_publication_operation(
                publication_id,
                OperationOutcome.FAILED,
                publication_state="rolled_back",
                publication_phase=terminal_phase,
            )

    @staticmethod
    def _has_compensation_applied_marker(publication: dict[str, Any]) -> bool:
        phases = publication["receipt"].get("phases", [])
        if not isinstance(phases, list):
            raise ValidationError("runtime publication receipt phases must be a list")
        markers = [
            phase
            for phase in phases
            if isinstance(phase, dict)
            and phase.get("phase") == "compensation_applied"
        ]
        if not markers:
            return False
        if len(markers) != 1 or str(markers[0].get("pid") or "") != str(
            publication["pid"]
        ):
            raise ValidationError(
                "invalid process exec compensation-applied marker: "
                f"{publication['publication_id']}"
            )
        return True

    def _record_exec_recovery_failure(
        self,
        claimed: dict[str, Any],
        *,
        recovery_lease_id: str,
        error: Exception,
    ) -> None:
        publication_id = str(claimed["publication_id"])
        with self._unit_of_work.transaction():
            if not self._publications.advance_runtime_publication(
                publication_id,
                state="failed",
                phase="startup_compensation_failed",
                error={
                    "code": "process_exec_compensation_failed",
                    "error_type": type(error).__name__,
                },
                expected_states={"rollback_pending"},
                recovery_lease_id=recovery_lease_id,
            ):
                raise ValidationError(
                    "process exec recovery lease changed before failure publication: "
                    f"{publication_id}"
                ) from error
            self._reconcile_publication_operation(
                publication_id,
                OperationOutcome.UNKNOWN,
                publication_state="failed",
                publication_phase="startup_compensation_failed",
            )

    def _fail_closed_manual_exec_publication(
        self,
        publication: dict[str, Any],
    ) -> None:
        with self._unit_of_work.transaction():
            self._reconcile_publication_operation(
                publication["publication_id"],
                OperationOutcome.UNKNOWN,
                publication_state="manual",
                publication_phase=str(publication["phase"]),
            )
        raise ValidationError(
            "process exec publication requires manual recovery: "
            f"{publication['publication_id']}"
        )

    def _require_exec_publication_resolved(self, publication_id: str) -> None:
        current = self._publications.get_runtime_publication(publication_id)
        if current is None or current["state"] in {"committed", "rolled_back"}:
            return
        raise ValidationError(
            f"cannot claim unresolved process exec publication: {publication_id}"
        )

    @staticmethod
    def _recovery_lease_id(publication: dict[str, Any]) -> str:
        recovery = dict(publication["receipt"].get("recovery") or {})
        recovery_lease_id = str(recovery.get("lease_id") or "")
        if not recovery_lease_id:
            raise ValidationError(
                f"process exec recovery claim has no lease: {publication['publication_id']}"
            )
        return recovery_lease_id

    def _exec_rollback_state(self, publication: dict[str, Any]) -> ExecRollbackState:
        plan = publication["plan"]
        snapshot = SnapshotCodec.decode_mapping(dict(plan["before_snapshot"]))
        return ExecRollbackState(
            snapshot=snapshot,
            tool_ids=frozenset(str(item) for item in plan.get("before_tool_ids", [])),
            tool_handles=self._tools.reconstruct_persisted_jit_handles(
                dict(snapshot.jit_sources)
            ),
            capability_rollback_token=publication["publication_id"],
        )

    @staticmethod
    def _exec_admission_token(
        publication: dict[str, Any],
    ) -> ProcessExecutionToken:
        plan = dict(publication.get("plan") or {})
        generation = plan.get("admission_execution_generation")
        owner_id = str(plan.get("admission_execution_owner_id") or "")
        lease_id = str(plan.get("admission_execution_lease_id") or "")
        pid = str(publication.get("pid") or "")
        if (
            not pid
            or isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
            or not owner_id
            or not lease_id
        ):
            raise ValidationError(
                "process exec publication has invalid admission token: "
                f"{publication.get('publication_id')}"
            )
        return ProcessExecutionToken(
            pid=pid,
            generation=generation,
            owner_id=owner_id,
            lease_id=lease_id,
        )

    def cleanup_failed_launch_artifacts(self, publication: dict[str, Any]) -> None:
        """Compensate only artifacts owned by one durable publication."""

        self._cleanup_publication_artifacts(
            publication,
            reason="process_launch_publication_compensation",
        )
        if self._processes.get_process(str(publication["pid"])) is None:
            self.assert_publication_artifacts_removed(publication)

    def _cleanup_publication_artifacts(
        self,
        publication: dict[str, Any],
        *,
        reason: str,
    ) -> None:
        publication_id = str(publication["publication_id"])
        pid = str(publication["pid"])
        plan = dict(publication.get("plan") or {})
        artifacts = list((publication.get("receipt") or {}).get("artifacts") or [])
        seen: set[str] = set()
        for artifact in reversed(artifacts):
            if not isinstance(artifact, dict):
                raise ValidationError(
                    f"invalid runtime publication artifact receipt: {publication_id}"
                )
            artifact_id = str(artifact.get("artifact_id") or "")
            if not artifact_id or artifact_id in seen:
                continue
            seen.add(artifact_id)
            self._cleanup_publication_artifact(
                publication_id,
                pid,
                artifact,
                publication_kind=str(publication["kind"]),
                reason=reason,
            )

        planned_workspace = plan.get("materialized_workspace_root")
        if planned_workspace and not any(
            isinstance(artifact, dict)
            and artifact.get("kind") == "workspace"
            and artifact.get("path") == planned_workspace
            for artifact in artifacts
        ):
            self._package_installer.cleanup_publication_workspace(
                str(planned_workspace),
                reason=reason,
            )

    def _cleanup_publication_artifact(
        self,
        publication_id: str,
        pid: str,
        artifact: dict[str, Any],
        *,
        publication_kind: str,
        reason: str,
    ) -> None:
        kind = str(artifact.get("kind") or "")
        handlers = {
            "loaded_skill": self._cleanup_loaded_skill_artifact,
            "tool": self._cleanup_tool_artifact,
            "tool_candidate": self._cleanup_tool_candidate_artifact,
            "capability": self._cleanup_capability_artifact,
            "workspace": self._cleanup_workspace_artifact,
        }
        handler = handlers.get(kind)
        if handler is None:
            raise ValidationError(
                f"no compensation handler for publication artifact kind: {kind or '<missing>'}"
            )
        handler(
            publication_id,
            pid,
            artifact,
            publication_kind=publication_kind,
            reason=reason,
        )

    def _cleanup_loaded_skill_artifact(
        self,
        _publication_id: str,
        _pid: str,
        _artifact: dict[str, Any],
        *,
        publication_kind: str,
        reason: str,
    ) -> None:
        # The enclosing process snapshot/launch row owns the loaded mapping.
        del publication_kind, reason

    def _cleanup_tool_artifact(
        self,
        _publication_id: str,
        pid: str,
        artifact: dict[str, Any],
        *,
        publication_kind: str,
        reason: str,
    ) -> None:
        del reason
        tool_id = str(artifact.get("tool_id") or "")
        if not tool_id:
            raise ValidationError("tool publication artifact has no tool_id")
        if self._processes.tool_id_referenced_outside_process(
            tool_id,
            excluding_pid=pid,
        ):
            raise ValidationError(
                f"publication-owned tool escaped process scope: {tool_id}"
            )
        process = self._processes.get_process(pid)
        defer_binding_cleanup = bool(
            process is not None
            and publication_kind == "process_launch"
            and process.status in {
                ProcessStatus.EXITED,
                ProcessStatus.FAILED,
                ProcessStatus.KILLED,
            }
        )
        if process is not None:
            binding_groups = (
                {
                    name: bound_tool_id
                    for name, bound_tool_id in process.tool_table.items()
                    if bound_tool_id == tool_id
                },
                {
                    name: bound_tool_id
                    for name, bound_tool_id in process.model_tool_table.items()
                    if bound_tool_id == tool_id
                },
            )
            if not defer_binding_cleanup:
                for bindings in binding_groups:
                    if bindings:
                        self._processes.remove_process_tool_bindings(pid, bindings)
        self._tools.forget_loaded_jit(tool_id)
        self._extensions.delete_tool(tool_id)

    def _cleanup_tool_candidate_artifact(
        self,
        _publication_id: str,
        pid: str,
        artifact: dict[str, Any],
        *,
        publication_kind: str,
        reason: str,
    ) -> None:
        candidate_id = str(artifact.get("candidate_id") or "")
        if not candidate_id:
            raise ValidationError(
                "tool candidate publication artifact has no candidate_id"
            )
        descriptor_oid = self._candidate_descriptor_oid(artifact)
        candidate = self._extensions.get_tool_candidate(candidate_id)
        if (
            descriptor_oid is None
            and candidate is not None
            and candidate.registered_tool_id is None
        ):
            raise ValidationError(
                "unregistered tool candidate receipt cannot omit its descriptor: "
                f"{candidate_id}"
            )
        process = self._processes.get_process(pid)
        if (
            candidate is not None
            and process is not None
            and publication_kind == "process_launch"
            and process.status
            in {ProcessStatus.EXITED, ProcessStatus.FAILED, ProcessStatus.KILLED}
            and candidate.registered_tool_id
            in {*process.tool_table.values(), *process.model_tool_table.values()}
        ):
            # Terminal launch rows are immutable. The strict launch cleanup
            # invokes artifact compensation again after deleting that row.
            return
        if candidate is not None or (
            descriptor_oid is not None
            and self._objects.get_object(descriptor_oid) is not None
        ):
            self._tools.discard_candidate(
                pid,
                candidate_id,
                descriptor_oid=descriptor_oid,
                exact_descriptor=True,
                discarded_by="runtime.publication",
                reason=reason,
            )

    def _cleanup_capability_artifact(
        self,
        publication_id: str,
        _pid: str,
        artifact: dict[str, Any],
        *,
        publication_kind: str,
        reason: str,
    ) -> None:
        del reason
        cap_id = str(artifact.get("capability_id") or "")
        if not cap_id:
            raise ValidationError(
                "capability publication artifact has no capability_id"
            )
        capability = self._authority.get_capability(cap_id)
        if (
            capability is not None
            and publication_kind == "process_exec"
            and capability.metadata.get("runtime_publication_id") != publication_id
        ):
            raise ValidationError(f"capability receipt ownership mismatch: {cap_id}")
        self._delete_publication_capability(cap_id)

    def _cleanup_workspace_artifact(
        self,
        _publication_id: str,
        _pid: str,
        artifact: dict[str, Any],
        *,
        publication_kind: str,
        reason: str,
    ) -> None:
        del publication_kind
        workspace_root = str(artifact.get("path") or "")
        if not workspace_root:
            raise ValidationError("workspace publication artifact has no path")
        self._package_installer.cleanup_publication_workspace(
            workspace_root,
            reason=reason,
        )

    def _delete_publication_capability(self, cap_id: str) -> None:
        self._authority.delete_publication_capability(cap_id)

    @staticmethod
    def _publication_tool_artifact_ids(
        artifacts: Iterable[Any],
    ) -> tuple[str, ...]:
        tool_ids: list[str] = []
        for artifact in artifacts:
            if not isinstance(artifact, dict) or artifact.get("kind") != "tool":
                continue
            tool_id = artifact.get("tool_id")
            if not isinstance(tool_id, str) or not tool_id or "\x00" in tool_id:
                raise ValidationError("tool publication artifact has no valid tool_id")
            tool_ids.append(tool_id)
        return tuple(tool_ids)

    @staticmethod
    def _candidate_descriptor_oid(artifact: dict[str, Any]) -> str | None:
        descriptor_state = artifact.get("descriptor_state")
        if descriptor_state not in {"object", "not_created"}:
            raise ValidationError(
                "tool candidate publication artifact has invalid descriptor state"
            )
        if "descriptor_oid" not in artifact:
            raise ValidationError(
                "tool candidate publication artifact has no descriptor identity"
            )
        descriptor_oid = artifact["descriptor_oid"]
        if descriptor_state == "not_created" and descriptor_oid is None:
            return None
        if descriptor_state == "not_created":
            raise ValidationError(
                "tool candidate publication artifact has inconsistent descriptor state"
            )
        if (
            not isinstance(descriptor_oid, str)
            or not descriptor_oid
            or "\x00" in descriptor_oid
        ):
            raise ValidationError(
                "tool candidate publication artifact has invalid descriptor identity"
            )
        return descriptor_oid

    def assert_publication_artifacts_removed(
        self,
        publication: dict[str, Any],
    ) -> None:
        """Fail closed unless every exact compensation identity converged."""

        publication_id = str(publication["publication_id"])
        pid = str(publication["pid"])
        artifacts = list((publication.get("receipt") or {}).get("artifacts") or [])
        tool_rows = self._extensions.get_existing_tool_ids(
            self._publication_tool_artifact_ids(artifacts)
        )
        process = self._processes.get_process(pid)
        leftovers = [
            artifact_id
            for artifact in artifacts
            if (
                artifact_id := self._publication_artifact_leftover(
                    artifact,
                    process=process,
                    tool_rows=tool_rows,
                )
            )
        ]
        if leftovers:
            raise ValidationError(
                "runtime publication compensation did not converge: "
                f"{publication_id}: {sorted(set(leftovers))}"
            )

    def _publication_artifact_leftover(
        self,
        artifact: Any,
        *,
        process: Any | None,
        tool_rows: Collection[str],
    ) -> str | None:
        if not isinstance(artifact, dict):
            return "invalid_receipt"
        kind = str(artifact.get("kind") or "")
        artifact_id = str(artifact.get("artifact_id") or kind or "invalid_receipt")
        checks = {
            "capability": self._capability_artifact_remains,
            "tool": self._tool_artifact_remains,
            "tool_candidate": self._tool_candidate_artifact_remains,
            "loaded_skill": self._loaded_skill_artifact_remains,
            "workspace": self._workspace_artifact_remains,
        }
        check = checks.get(kind)
        if check is None:
            return artifact_id
        return artifact_id if check(artifact, process, tool_rows) else None

    def _capability_artifact_remains(
        self,
        artifact: dict[str, Any],
        _process: Any | None,
        _tool_rows: Collection[str],
    ) -> bool:
        return self._authority.get_capability(
            str(artifact.get("capability_id") or "")
        ) is not None

    def _tool_artifact_remains(
        self,
        artifact: dict[str, Any],
        process: Any | None,
        tool_rows: Collection[str],
    ) -> bool:
        tool_id = str(artifact.get("tool_id") or "")
        aliases = set()
        if process is not None:
            aliases.update(process.tool_table.values())
            aliases.update(process.model_tool_table.values())
        return any(
            (
                tool_id in tool_rows,
                self._tools.loaded_tool_handle(tool_id) is not None,
                self._tools.jit_source(tool_id) is not None,
                tool_id in aliases,
            )
        )

    def _tool_candidate_artifact_remains(
        self,
        artifact: dict[str, Any],
        _process: Any | None,
        _tool_rows: Collection[str],
    ) -> bool:
        candidate_id = str(artifact.get("candidate_id") or "")
        descriptor_oid = self._candidate_descriptor_oid(artifact)
        candidate = self._extensions.get_tool_candidate(candidate_id)
        if (
            descriptor_oid is None
            and candidate is not None
            and candidate.registered_tool_id is None
        ):
            raise ValidationError(
                "unregistered tool candidate receipt cannot omit its descriptor: "
                f"{candidate_id}"
            )
        return (
            candidate is not None
            or (
                descriptor_oid is not None
                and self._objects.get_object(descriptor_oid) is not None
            )
        )

    @staticmethod
    def _loaded_skill_artifact_remains(
        artifact: dict[str, Any],
        process: Any | None,
        _tool_rows: Collection[str],
    ) -> bool:
        if process is None:
            return False
        loaded = process.loaded_skills.get(str(artifact.get("skill_id") or ""))
        if loaded is None:
            return False
        expected_loaded_at = artifact.get("loaded_at")
        return expected_loaded_at is None or (
            isinstance(loaded, dict) and loaded.get("loaded_at") == expected_loaded_at
        )

    def _workspace_artifact_remains(
        self,
        artifact: dict[str, Any],
        _process: Any | None,
        _tool_rows: Collection[str],
    ) -> bool:
        return self._package_installer.publication_workspace_exists(
            str(artifact.get("path") or "")
        )

    def _advance_publication(self, publication_id: str, phase: str) -> None:
        if not self._publications.advance_runtime_publication(
            publication_id,
            state="applying",
            phase=phase,
            receipt={"phase": phase},
            expected_states={"planning", "applying"},
        ):
            raise ValidationError(f"process exec publication changed during {phase}: {publication_id}")

    def _reconcile_publication_operation(
        self,
        publication_id: str,
        outcome: OperationOutcome,
        *,
        publication_state: str,
        publication_phase: str,
    ) -> None:
        publication = self._publications.get_runtime_publication(publication_id)
        if publication is None:
            raise ValidationError(
                f"runtime publication disappeared during operation reconciliation: {publication_id}"
            )
        plan = publication["plan"]
        operation_id = plan.get("operation_id")
        if publication["kind"] != "process_exec":
            raise ValidationError(
                "process exec operation reconciliation received publication kind: "
                f"{publication['kind']}"
            )
        if publication["state"] != publication_state or publication["phase"] != publication_phase:
            raise ValidationError(
                "process exec publication changed during operation reconciliation: "
                f"{publication_id}"
            )
        if not operation_id:
            reverse_bindings = (
                self._operations.runtime_publication_binding_operation_ids(
                    publication_id
                )
            )
            if plan.get("operation_binding_version") is not None or reverse_bindings:
                raise ValidationError(
                    "process exec publication lost its durable operation binding: "
                    f"{publication_id} -> {reverse_bindings or '<missing>'}"
                )
            if not self._publications.mark_runtime_publication_operation_reconciled(
                publication_id,
                expected_kind="process_exec",
                expected_state=publication_state,
                expected_phase=publication_phase,
                expected_operation_id=None,
            ):
                raise ValidationError(
                    "process exec publication changed while marking unbound reconciliation: "
                    f"{publication_id}"
                )
            return
        publication_pid = str(publication["pid"])
        if str(publication["plan"].get("pid") or "") != publication_pid:
            raise ValidationError(
                "process exec publication plan PID does not match its durable PID: "
                f"{publication_id}"
            )
        self._operations.reconcile_runtime_publication(
            str(operation_id),
            outcome,
            publication_id=publication_id,
            publication_kind="process_exec",
            publication_state=publication_state,
            publication_phase=publication_phase,
            expected_kind="runtime",
            expected_name="process.exec",
            expected_actor=publication_pid,
            expected_pid=publication_pid,
        )

    def reconcile_terminal_publications(self) -> list[str]:
        """Converge operation outcomes for already-terminal exec receipts."""

        reconciled: list[str] = []
        outcomes = {
            "committed": OperationOutcome.SUCCEEDED,
            "rolled_back": OperationOutcome.FAILED,
            "failed": OperationOutcome.UNKNOWN,
            "manual": OperationOutcome.UNKNOWN,
        }
        for state, outcome in outcomes.items():
            after: RuntimePublicationCursor | None = None
            while True:
                page = self._publications.query_runtime_publication_operation_reconciliation(
                    kind="process_exec",
                    state=state,
                    after=after,
                    limit=self._reconciliation_page_size,
                )
                previous = after
                for publication in page.records:
                    cursor = RuntimePublicationCursor(
                        publication["created_at"],
                        publication["publication_id"],
                    )
                    if (
                        publication["kind"] != "process_exec"
                        or publication["state"] != state
                        or publication["operation_reconciled"]
                        or (previous is not None and cursor <= previous)
                    ):
                        raise ValidationError(
                            "runtime publication repository returned an invalid exec reconciliation page"
                        )
                    with self._unit_of_work.transaction():
                        self._reconcile_publication_operation(
                            publication["publication_id"],
                            outcome,
                            publication_state=state,
                            publication_phase=publication["phase"],
                        )
                    if len(reconciled) < self._reconciliation_page_size:
                        reconciled.append(publication["publication_id"])
                    previous = cursor
                if page.next_cursor is None:
                    break
                if previous is None or page.next_cursor != previous:
                    raise ValidationError(
                        "runtime publication repository returned an invalid exec reconciliation cursor"
                    )
                after = page.next_cursor
        return reconciled

    def preflight_id(self, image_id: str) -> None:
        self.preflight(self._launch.require_image(image_id))

    def preflight_exec(self, image_id: str) -> None:
        """Validate the complete replacement image before opening evidence."""

        self.preflight_id(image_id)

    def preflight(self, image: AgentImage) -> None:
        self._require_image_modules(image)
        boot_kind = str(image.boot.get("kind", "fresh"))
        if boot_kind == "checkpoint_commit":
            self._checkpoint_installer.preflight(image)
        elif boot_kind == "image_package":
            self._package_installer.preflight(image)

    def configure_spawn(
        self,
        pid: str,
        image_id: str,
        publication_id: str,
    ) -> None:
        try:
            image = self._launch.require_image(image_id)
            boot_kind = str(image.boot.get("kind", "fresh"))
            self.preflight(image)
            workspace_root = (
                self._package_installer.planned_workspace_root(
                    pid,
                    image,
                    materialization_id=publication_id,
                )
                if boot_kind == "image_package"
                else None
            )
            if not self._publications.update_runtime_publication_plan(
                publication_id,
                {
                    "boot_kind": boot_kind,
                    "materialized_workspace_root": workspace_root,
                },
                expected_states={"planning", "applying"},
            ):
                raise ValidationError(
                    f"process launch publication changed before image boot: {publication_id}"
                )
            assigned_by = f"publication:{publication_id}"
            initial_tool_table, initial_model_tool_table = self._configure_tools(
                pid,
                image,
                assigned_by,
            )
            self._instantiate_boot(
                pid,
                image,
                boot_kind,
                workspace_root=workspace_root,
                publication_id=publication_id,
            )
            process = self._processes.get_process(pid)
        except Exception as exc:
            self._fail_boot(pid, image_id, exc, phase="process.spawn")
            raise
        try:
            self._configure_skills(
                pid,
                image,
                assigned_by,
                publication_id=publication_id,
                initial_tool_table=initial_tool_table,
                initial_model_tool_table=initial_model_tool_table,
            )
        except Exception as exc:
            if boot_kind == "image_package":
                self._package_installer.cleanup(
                    pid,
                    image,
                    reason="image_package_default_skills_failed",
                )
            self._fail_boot(pid, image_id, exc, phase="image.default_skills")
            raise
        self._record_spawn_authority(
            pid,
            image,
            process,
            boot_kind,
            publication_id=publication_id,
        )

    def _configure_tools(
        self,
        pid: str,
        image: AgentImage,
        assigned_by: str,
    ) -> tuple[dict[str, str], dict[str, str]]:
        full_table = self._tools.configure_process_tools(
            pid,
            sorted(image.default_tools),
            assigned_by=assigned_by,
        )
        model_table = self._tools.configure_model_tool_projection(
            pid,
            sorted(self._tools.initial_tool_projection(image)),
            assigned_by=f"{assigned_by}:model_projection",
        )
        return full_table, model_table

    def _configure_skills(
        self,
        pid: str,
        image: AgentImage,
        assigned_by: str,
        *,
        publication_id: str | None = None,
        initial_tool_table: Mapping[str, str],
        initial_model_tool_table: Mapping[str, str],
    ) -> None:
        process = self._processes.get_process(pid)
        if process is None:
            return
        self._apply_loaded_skill_tool_table(
            process,
            initial_tool_table=initial_tool_table,
            initial_model_tool_table=initial_model_tool_table,
        )
        for skill_id in image.default_skills:
            current = self._processes.get_process(pid)
            if current is not None and skill_id in current.loaded_skills:
                continue
            self._skills.activate_skill(
                pid,
                skill_id,
                actor=assigned_by,
                require_capability=False,
                publication_id=publication_id,
            )

    def _apply_loaded_skill_tool_table(
        self,
        process: Any,
        *,
        initial_tool_table: Mapping[str, str],
        initial_model_tool_table: Mapping[str, str],
    ) -> None:
        if not process.loaded_skills:
            return
        updated = dict(process.tool_table)
        updated_model = dict(process.model_tool_table)
        updated_loaded = dict(process.loaded_skills)
        for skill_id, loaded in list(process.loaded_skills.items()):
            if not isinstance(loaded, dict):
                continue
            activation_kind = loaded.get("activation_kind", "registered")
            if activation_kind == "builtin_projection":
                if not self._skills.builtin_projection_supported_by_process(
                    process,
                    str(skill_id),
                    loaded,
                ):
                    # Fork/exec may target a narrower image. Built-in Skills
                    # are visibility-only and must never carry parent bindings
                    # into the target image's complete tool table.
                    updated_loaded.pop(skill_id, None)
                    self._audit.record(
                        actor="runtime",
                        action="skill.builtin_projection_dropped",
                        target=f"process:{process.pid}",
                        decision={
                            "skill_id": skill_id,
                            "image_id": process.image_id,
                            "authority_changed": False,
                            "reason": "target image does not contain the complete built-in Skill tool set",
                        },
                    )
                    continue
                rebased = self._rebase_builtin_loaded_skill(
                    loaded,
                    initial_model_tool_table=initial_model_tool_table,
                )
                updated_loaded[skill_id] = rebased
                updated_model.update(
                    self._loaded_string_bindings(rebased, ("tool_ids",))
                )
                continue
            if activation_kind != "registered":
                raise ValidationError(
                    f"unknown loaded Skill activation_kind during image boot: {activation_kind}"
                )
            rebased = self._rebase_registered_loaded_skill(
                str(skill_id),
                loaded,
                initial_tool_table=initial_tool_table,
                initial_model_tool_table=initial_model_tool_table,
            )
            updated_loaded[skill_id] = rebased
            bindings = self._loaded_string_bindings(
                rebased,
                ("tool_ids", "jit_tool_ids"),
            )
            updated.update(bindings)
            updated_model.update(bindings)
        process.tool_table = updated
        process.model_tool_table = updated_model
        process.loaded_skills = updated_loaded
        process.updated_at = utc_now()
        self._processes.patch_process(
            process.pid,
            {
                "tool_table": process.tool_table,
                "model_tool_table": process.model_tool_table,
                "loaded_skills": process.loaded_skills,
                "updated_at": process.updated_at,
            },
            expected_revision=process.revision,
        )

    def _rebase_builtin_loaded_skill(
        self,
        loaded: dict[str, Any],
        *,
        initial_model_tool_table: Mapping[str, str],
    ) -> dict[str, Any]:
        bindings = self._loaded_string_bindings(loaded, ("tool_ids",))
        rebased = dict(loaded)
        rebased["base_tool_ids"] = dict(bindings)
        rebased["base_model_tool_ids"] = {
            name: str(initial_model_tool_table[name])
            for name in bindings
            if name in initial_model_tool_table
        }
        return rebased

    def _rebase_registered_loaded_skill(
        self,
        skill_id: str,
        loaded: dict[str, Any],
        *,
        initial_tool_table: Mapping[str, str],
        initial_model_tool_table: Mapping[str, str],
    ) -> dict[str, Any]:
        if not all(
            isinstance(loaded.get(field), dict)
            for field in ("base_tool_ids", "base_model_tool_ids")
        ):
            raise ValidationError(
                "loaded Skill state is missing canonical tool provenance "
                f"during image boot: {skill_id}"
            )
        published_names = set(
            self._loaded_string_bindings(
                loaded,
                ("tool_ids", "jit_tool_ids"),
            )
        )
        # A fork or checkpoint-backed exec can carry ordinary loaded Skills
        # across images. Their saved base bindings describe the source image,
        # so retaining them would let a later unload resurrect a source-only
        # tool after the target image has been configured. Rebase only to the
        # exact target image tables captured before inherited Skills were
        # applied.
        rebased = dict(loaded)
        rebased["base_tool_ids"] = {
            name: str(initial_tool_table[name])
            for name in published_names
            if name in initial_tool_table
        }
        rebased["base_model_tool_ids"] = {
            name: str(initial_model_tool_table[name])
            for name in published_names
            if name in initial_model_tool_table
        }
        return rebased

    @staticmethod
    def _loaded_string_bindings(
        loaded: Mapping[str, Any],
        fields: Iterable[str],
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        for field in fields:
            mapping = loaded.get(field)
            if not isinstance(mapping, dict):
                continue
            result.update(
                {
                    name: tool_id
                    for name, tool_id in mapping.items()
                    if isinstance(name, str) and isinstance(tool_id, str)
                }
            )
        return result

    def _instantiate_boot(
        self,
        pid: str,
        image: AgentImage,
        boot_kind: str,
        *,
        workspace_root: str | None = None,
        publication_id: str | None = None,
    ) -> None:
        if boot_kind == "checkpoint_commit":
            self._checkpoint_installer.install(
                pid,
                image,
                publication_id=publication_id,
            )
        elif boot_kind == "image_package":
            self._package_installer.install(
                pid,
                image,
                workspace_root=workspace_root,
                publication_id=publication_id,
            )

    def _require_image_modules(self, image: AgentImage) -> None:
        missing: list[dict[str, str]] = []
        for module in image.required_modules:
            module_id = str(module.get("module_id", ""))
            source_sha256 = str(module.get("source_sha256", ""))
            if module_id and not self._modules.is_loaded(
                module_id,
                source_sha256 or None,
            ):
                missing.append(
                    {
                        "module_id": module_id,
                        "source_sha256": source_sha256,
                    }
                )
        if missing:
            raise ValidationError(
                f"image requires startup modules that are not loaded: {missing}"
            )

    def _fail_boot(
        self,
        pid: str,
        image_id: str,
        exc: Exception,
        *,
        phase: str,
    ) -> None:
        process = self._processes.get_process(pid)
        if process is not None:
            self._process.transitions.transition(
                pid,
                ProcessStatus.FAILED,
                expected_revision=process.revision,
                expected_status=process.status,
                expected_state_generation=process.state_generation,
                outcome=FailedProcessOutcome(code=f"image_boot_{phase}_failed"),
                status_message=str(exc),
            )
        self._audit.record(
            actor="runtime",
            action="image.boot.failed",
            target=f"process:{pid}",
            decision={"image": image_id, "phase": phase, "error": str(exc)},
        )

    def _record_spawn_authority(
        self,
        pid: str,
        image: AgentImage,
        process: Any,
        boot_kind: str,
        publication_id: str | None = None,
    ) -> None:
        if process is not None:
            self._checkpoint.grant_process_defaults(
                pid,
                issued_by=f"image:{image.image_id}",
                publication_id=publication_id,
            )
        if process is not None and process.parent_pid is not None:
            self._audit.record(
                actor="runtime",
                action="image.default_capability_skipped_for_child",
                target=f"process:{pid}",
                decision={
                    "image": image.image_id,
                    "parent_pid": process.parent_pid,
                },
            )
            return
        manifest = self._authority_manifests.summary_for_process(pid)
        self._audit.record(
            actor="runtime",
            action="image.required_capabilities_declared_only",
            target=f"process:{pid}",
            decision={
                "image": image.image_id,
                "boot_kind": boot_kind,
                "required_capabilities": len(image.required_capabilities),
                "authority_manifest_id": (
                    manifest.get("manifest_id") if manifest else None
                ),
                "missing_required_capabilities": (
                    manifest.get("missing_required_capabilities", [])
                    if manifest
                    else list(image.required_capabilities)
                ),
                "reason": (
                    "image requirements are declarations; launch manifests own grants"
                ),
            },
        )


__all__ = ["ImageBootService"]
