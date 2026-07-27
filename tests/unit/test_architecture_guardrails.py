from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import check_architecture as checker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ALLOWLIST = PROJECT_ROOT / "scripts" / "architecture_allowlist.json"


def _write_source(root: Path, relative: str, source: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _write_current_baseline(root: Path) -> Path:
    path = root / "architecture_allowlist.json"
    path.write_text(
        json.dumps(checker.allowlist_for(checker.scan_architecture(root)), indent=2),
        encoding="utf-8",
    )
    return path


def _long_function(statement_count: int) -> str:
    return "def legacy_hotspot():\n" + "".join(
        f"    value_{index} = {index}\n" for index in range(statement_count)
    )


def _complex_function(branch_count: int) -> str:
    branches = "".join(
        f"    if value == {index}:\n        result += {index}\n"
        for index in range(branch_count)
    )
    return f"def legacy_hotspot(value):\n    result = 0\n{branches}    return result\n"


class TestArchitectureGuardrails:

    def test_default_function_line_limit_is_200(self) -> None:
        assert checker.MAX_FUNCTION_LINES == 200

    def test_checked_in_ratchet_accepts_the_current_tree(self) -> None:
        assert checker.check_architecture(PROJECT_ROOT, PROJECT_ALLOWLIST) == []

    @pytest.mark.parametrize(
        "package",
        [
            "models",
            "capability",
            "images",
            "primitives",
            "sdk",
            "human",
            "llm",
            "tools",
        ],
    )
    @pytest.mark.parametrize(
        "statement",
        [
            "from agent_libos.runtime.runtime import Runtime",
            "from ..runtime.runtime import Runtime",
            "from agent_libos import runtime",
        ],
    )
    def test_new_reverse_runtime_import_fails(
        self,
        tmp_path: Path,
        package: str,
        statement: str,
    ) -> None:
        relative = f"agent_libos/{package}/sample.py"
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)

        _write_source(
            tmp_path,
            relative,
            f"{statement}\n\nVALUE = 2\n",
        )

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("concrete-runtime-import" in error for error in errors)

    def test_lower_layer_api_import_fails(self, tmp_path: Path) -> None:
        relative = "agent_libos/tools/sample.py"
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)

        _write_source(
            tmp_path,
            relative,
            "from agent_libos.api.cli import main\n",
        )

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("concrete-api-import" in error for error in errors)

    def test_runtime_layer_api_import_fails(self, tmp_path: Path) -> None:
        relative = "agent_libos/runtime/service.py"
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)

        _write_source(
            tmp_path,
            relative,
            "from agent_libos.api.cli import main\n",
        )

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("concrete-api-import" in error for error in errors)

    def test_new_raw_asyncio_to_thread_dispatch_fails(self, tmp_path: Path) -> None:
        relative = "agent_libos/human/provider.py"
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)

        _write_source(
            tmp_path,
            relative,
            "import asyncio\n\nasync def invoke(provider):\n"
            "    return await asyncio.to_thread(provider)\n",
        )

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("untracked-blocking-dispatch" in error for error in errors)

    @pytest.mark.parametrize(
        "source",
        [
            (
                "import asyncio\n\nasync def invoke(provider):\n"
                "    loop = asyncio.get_running_loop()\n"
                "    return await loop.run_in_executor(None, provider)\n"
            ),
            (
                "async def invoke(loop, provider):\n"
                "    return await loop.run_in_executor(executor=None, func=provider)\n"
            ),
        ],
        ids=["positional", "keyword"],
    )
    def test_new_raw_default_executor_dispatch_fails(
        self,
        tmp_path: Path,
        source: str,
    ) -> None:
        relative = "agent_libos/human/provider.py"
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)

        _write_source(tmp_path, relative, source)

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("untracked-blocking-dispatch" in error for error in errors)

    @pytest.mark.parametrize(
        "statement",
        [
            "store.update_process(process)",
            "getattr(store, 'update_process')(process)",
        ],
    )
    def test_whole_process_write_fails(self, tmp_path: Path, statement: str) -> None:
        relative = "agent_libos/runtime/sample.py"
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)

        _write_source(
            tmp_path,
            relative,
            f"def mutate(store, process):\n    {statement}\n",
        )

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("whole-process-write" in error for error in errors)

    @pytest.mark.parametrize(
        "statement",
        [
            "cursor.execute('DELETE FROM processes')",
            "store.select_table_rows('processes')",
            "getattr(store, 'delete_table_rows')('processes')",
        ],
    )
    @pytest.mark.parametrize(
        "relative",
        [
            "agent_libos/runtime/process_manager.py",
            "agent_libos/runtime/resource_manager.py",
            "agent_libos/runtime/checkpoint_manager.py",
            "agent_libos/runtime/checkpoint_reconciliation.py",
            "agent_libos/runtime/checkpoint_image.py",
            "agent_libos/runtime/image_boot.py",
            "agent_libos/runtime/data_flow_manager.py",
            "agent_libos/runtime/snapshots/exec_state.py",
            "agent_libos/modules/registry.py",
            "agent_libos/process_transition.py",
        ],
    )
    def test_migrated_runtime_storage_sql_bypass_fails(
        self,
        tmp_path: Path,
        statement: str,
        relative: str,
    ) -> None:
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)

        _write_source(
            tmp_path,
            relative,
            f"def mutate(store, cursor):\n    {statement}\n",
        )

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("runtime-storage-sql-bypass" in error for error in errors)

    def test_jit_projection_raw_sql_fails_outside_storage_trust_boundary(
        self,
        tmp_path: Path,
    ) -> None:
        cases = (
            (
                "agent_libos/tools/sample.py",
                "cursor.execute('DELETE FROM tools WHERE tool_id = ?', (tool_id,))",
                "tools",
            ),
            (
                "agent_libos/runtime/sample.py",
                "cursor.executemany("
                "'INSERT INTO process_tool_bindings (pid, binding_kind, tool_name, tool_id) '"
                "'VALUES (?, ?, ?, ?)', rows)",
                "process_tool_bindings",
            ),
            (
                "modules/sample.py",
                "getattr(cursor, 'execute')("
                "f'UPDATE tools SET ephemeral = 0 WHERE tool_id = {tool_id!r}')",
                "tools",
            ),
        )
        for index, (relative, statement, table) in enumerate(cases):
            case_root = tmp_path / f"case-{index}"
            _write_source(case_root, relative, "VALUE = 1\n")
            allowlist = _write_current_baseline(case_root)
            _write_source(
                case_root,
                relative,
                "def mutate(cursor, tool_id, rows):\n"
                f"    {statement}\n",
            )

            errors = checker.check_architecture(case_root, allowlist)

            assert any(
                "runtime-storage-sql-bypass" in error
                and f"raw-jit-projection-sql:{table}" in error
                for error in errors
            ), (relative, errors)

    def test_jit_projection_sql_local_name_indirection_is_detected(
        self,
        tmp_path: Path,
    ) -> None:
        relative = "agent_libos/tools/sample.py"
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)
        _write_source(
            tmp_path,
            relative,
            "SQL = 'DELETE FROM tools WHERE tool_id = ?'\n"
            "def mutate(cursor, tool_id):\n"
            "    statement = SQL\n"
            "    cursor.execute(statement, (tool_id,))\n",
        )

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any(
            "runtime-storage-sql-bypass" in error
            and "raw-jit-projection-sql:tools" in error
            for error in errors
        )

    @pytest.mark.parametrize(
        "relative",
        [
            "agent_libos/storage/sql.py",
            "agent_libos/storage/migrations/v4.py",
            "tests/storage/test_projection_fixture.py",
        ],
        ids=["storage-owner", "migration", "test-fixture"],
    )
    def test_jit_projection_raw_sql_is_limited_to_explicit_trust_boundary(
        self,
        tmp_path: Path,
        relative: str,
    ) -> None:
        _write_source(
            tmp_path,
            relative,
            "def migrate(cursor):\n"
            "    cursor.execute('UPDATE tools SET ephemeral = 0')\n"
            "    cursor.execute('DELETE FROM process_tool_bindings')\n",
        )

        report = checker.scan_architecture(tmp_path)

        assert not any(
            violation.rule == "runtime-storage-sql-bypass"
            and violation.detail.startswith("raw-jit-projection-sql:")
            for violation in report.violations
        )

    @pytest.mark.parametrize(
        ("repository_class", "statement"),
        [
            (
                "ProcessRepository",
                "self._delegate('get_process', pid)",
            ),
            (
                "ProcessRepository",
                "getattr(self._process_backend, 'get_process')(pid)",
            ),
            (
                "PayloadRetentionRepository",
                "self._delegate('scan_llm_call_payloads_for_retention')",
            ),
            (
                "PayloadRetentionRepository",
                "getattr(self._retention_backend, 'scan_llm_call_payloads_for_retention')()",
            ),
        ],
    )
    def test_migrated_repository_reflection_bypass_fails(
        self,
        tmp_path: Path,
        repository_class: str,
        statement: str,
    ) -> None:
        relative = "agent_libos/storage/repositories.py"
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)

        _write_source(
            tmp_path,
            relative,
            f"class {repository_class}:\n"
            "    def get_process(self, pid=None):\n"
            f"        return {statement}\n",
        )

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("typed-storage-reflection-bypass" in error for error in errors)

    @pytest.mark.parametrize(
        "source",
        [
            (
                "class CheckpointRestoreReconciler:\n"
                "    def __init__(self, store: Any, writer: Any, operations: Any):\n"
                "        self._store = store\n"
            ),
            (
                "class CheckpointRestoreReconciler:\n"
                "    def __init__(self, store: CheckpointRestorePublicationReader, "
                "writer: CheckpointRestorePublicationWriterPort, "
                "operations: RuntimePublicationOperationPort):\n"
                "        self._operations = operations\n"
                "    def read(self):\n"
                "        return self._operations.store.get_operation('operation')\n"
            ),
        ],
        ids=["untyped-constructor", "nested-operation-store"],
    )
    def test_checkpoint_reconciliation_requires_exact_typed_ports(
        self,
        tmp_path: Path,
        source: str,
    ) -> None:
        relative = "agent_libos/runtime/checkpoint_reconciliation.py"
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)
        _write_source(tmp_path, relative, source)

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("typed-storage-reflection-bypass" in error for error in errors)

    @pytest.mark.parametrize(
        ("relative", "source"),
        [
            (
                "agent_libos/runtime/checkpoint_manager.py",
                "class CheckpointManager:\n"
                "    def __init__(self, checkpoint_publication_writer: Any):\n"
                "        self._writer = checkpoint_publication_writer\n",
            ),
            (
                "agent_libos/runtime/builder.py",
                "def configure(host):\n"
                "    return CheckpointManager(\n"
                "        checkpoint_publication_writer=host.uow.publications,\n"
                "    )\n",
            ),
            (
                "agent_libos/runtime/checkpoint_manager.py",
                "class CheckpointManager:\n"
                "    def __init__(self, "
                "checkpoint_publication_writer: "
                "CheckpointRestorePublicationWriterPort, operations: Any):\n"
                "        self._operations = operations\n",
            ),
        ],
        ids=[
            "untyped-manager-port",
            "miswired-generic-publications",
            "untyped-operation-port",
        ],
    )
    def test_checkpoint_manager_requires_the_dedicated_typed_writer(
        self,
        tmp_path: Path,
        relative: str,
        source: str,
    ) -> None:
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)
        _write_source(tmp_path, relative, source)

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("typed-storage-reflection-bypass" in error for error in errors)

    @pytest.mark.parametrize(
        "source",
        [
            (
                "from agent_libos.storage import RuntimeStore\n"
                "class CheckpointManager:\n"
                "    def __init__(self, store: RuntimeStore):\n"
                "        self.store = store\n"
            ),
            (
                "from agent_libos.storage import UnitOfWork\n"
                "class CheckpointManager:\n"
                "    def __init__(self, unit_of_work: UnitOfWork):\n"
                "        self._store = unit_of_work\n"
                "    def read(self):\n"
                "        return self._store.get_process('pid')\n"
            ),
            (
                "from typing import Any\n"
                "class CheckpointManager:\n"
                "    def __init__(self, backend: Any):\n"
                "        self._backend = backend\n"
                "    def read(self):\n"
                "        return self._backend.get_process('pid')\n"
            ),
            (
                "from typing import Any\n"
                "from agent_libos.storage import UnitOfWork\n"
                "class CheckpointManager:\n"
                "    def __init__(self, unit_of_work: UnitOfWork, "
                "snapshot_repository: Any):\n"
                "        self._snapshot_rows = snapshot_repository\n"
            ),
            (
                "class CheckpointManager:\n"
                "    def run(self):\n"
                "        with self._snapshot_rows.transaction():\n"
                "            pass\n"
            ),
        ],
        ids=[
            "raw-runtime-store",
            "retained-store-alias",
            "any-backend-alias",
            "overridable-snapshot-repository",
            "snapshot-cursor-transaction",
        ],
    )
    def test_checkpoint_manager_cannot_retain_a_raw_store(
        self,
        tmp_path: Path,
        source: str,
    ) -> None:
        relative = "agent_libos/runtime/checkpoint_manager.py"
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)
        _write_source(tmp_path, relative, source)

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("typed-storage-reflection-bypass" in error for error in errors)

    def test_checkpoint_manager_builder_requires_unit_of_work_wiring(
        self,
        tmp_path: Path,
    ) -> None:
        relative = "agent_libos/runtime/builder.py"
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)
        _write_source(
            tmp_path,
            relative,
            "def configure(host, store):\n"
            "    return CheckpointManager(\n"
            "        store,\n"
            "        checkpoint_publication_writer="
            "host.uow.checkpoint_restore_publications,\n"
            "    )\n",
        )

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("typed-storage-reflection-bypass" in error for error in errors)

    @pytest.mark.parametrize(
        "statement",
        [
            "self.store.patch_process('pid', {}, expected_revision=1)",
            "self.store.append_process_memory_roots('pid', [])",
        ],
        ids=["patch-process", "append-memory-root"],
    )
    def test_runtime_facade_cannot_mutate_through_raw_host_store(
        self,
        tmp_path: Path,
        statement: str,
    ) -> None:
        relative = "agent_libos/runtime/runtime.py"
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)
        _write_source(
            tmp_path,
            relative,
            "class Runtime:\n"
            "    def add_handle_to_process_view(self):\n"
            f"        {statement}\n",
        )

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("typed-storage-reflection-bypass" in error for error in errors)

    def test_runtime_handle_publication_must_delegate_to_process_manager(
        self,
        tmp_path: Path,
    ) -> None:
        relative = "agent_libos/runtime/runtime.py"
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)
        _write_source(
            tmp_path,
            relative,
            "class Runtime:\n"
            "    def _add_handle_to_process_view(self, pid, handle):\n"
            "        self.uow.processes.append_process_memory_roots(pid, [handle])\n",
        )

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("typed-storage-reflection-bypass" in error for error in errors)

    @pytest.mark.parametrize(
        ("relative", "source"),
        [
            (
                "agent_libos/tools/broker.py",
                "class ToolBroker:\n"
                "    def __init__(self, store):\n"
                "        self.store = store\n",
            ),
            (
                "agent_libos/tools/broker.py",
                "class ToolBroker:\n"
                "    def list(self):\n"
                "        return self.store.list_tools()\n",
            ),
            (
                "agent_libos/runtime/builder.py",
                "def configure(host, store):\n"
                "    return ToolBroker(store)\n",
            ),
            (
                "agent_libos/tools/broker.py",
                "from typing import Any\n"
                "class ToolBroker:\n"
                "    def __init__(self, unit_of_work: Any):\n"
                "        self.unit_of_work = unit_of_work\n",
            ),
            (
                "agent_libos/tools/broker.py",
                "class RuntimeStore: pass\n"
                "class ToolBroker:\n"
                "    def __init__(self, backend: RuntimeStore):\n"
                "        self.backend = backend\n"
                "    def list(self):\n"
                "        return self.backend.list_tools()\n",
            ),
            (
                "agent_libos/tools/broker.py",
                "class ToolBroker:\n"
                "    def read(self, pid):\n"
                "        return self.unit_of_work.processes.get_process(pid)\n",
            ),
            (
                "agent_libos/tools/broker.py",
                "class ToolBroker:\n"
                "    def read(self, tool_id):\n"
                "        return self.objects.get_tool_spec(tool_id)\n",
            ),
        ],
        ids=[
            "constructor-store",
            "retained-store",
            "builder-raw-store",
            "untyped-unit-of-work",
            "renamed-runtime-store",
            "nested-process-facade",
            "wrong-tool-facade",
        ],
    )
    def test_tool_broker_requires_unit_of_work_facades(
        self,
        tmp_path: Path,
        relative: str,
        source: str,
    ) -> None:
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)
        _write_source(tmp_path, relative, source)

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("typed-storage-reflection-bypass" in error for error in errors)

    @pytest.mark.parametrize(
        "statement",
        [
            "self.runtime.store.get_process(self.pid)",
            "self.runtime.store.get_capability(capability_id)",
        ],
        ids=["process-read", "capability-read"],
    )
    def test_syscall_session_cannot_read_through_raw_runtime_store(
        self,
        tmp_path: Path,
        statement: str,
    ) -> None:
        relative = "agent_libos/runtime/syscalls.py"
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)
        _write_source(
            tmp_path,
            relative,
            "class LibOSSyscallSession:\n"
            "    def inspect(self, capability_id):\n"
            f"        return {statement}\n",
        )

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("typed-storage-reflection-bypass" in error for error in errors)

    @pytest.mark.parametrize(
        "statement",
        [
            "operations.store.get_operation(operation_id)",
            "self.capabilities.store.list_capabilities(pid)",
            "self.store.get_operation(operation_id)",
            "self.store.list_capabilities(pid)",
        ],
        ids=[
            "nested-operation-store",
            "nested-capability-store",
            "process-facade-operation",
            "process-facade-capability",
        ],
    )
    def test_process_manager_launch_recovery_uses_exact_typed_facades(
        self,
        tmp_path: Path,
        statement: str,
    ) -> None:
        relative = "agent_libos/runtime/process_manager.py"
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)
        _write_source(
            tmp_path,
            relative,
            "class ProcessManager:\n"
            "    def recover(self, operations, operation_id, pid):\n"
            f"        return {statement}\n",
        )

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("typed-storage-reflection-bypass" in error for error in errors)

    def test_runtime_module_concrete_runtime_import_fails(
        self,
        tmp_path: Path,
    ) -> None:
        relative = "modules/example/module.py"
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)

        _write_source(
            tmp_path,
            relative,
            "from agent_libos.runtime.runtime import Runtime\n",
        )

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("concrete-runtime-import" in error for error in errors)

    def test_runtime_module_private_component_access_fails(
        self,
        tmp_path: Path,
    ) -> None:
        relative = "modules/example/module.py"
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)

        _write_source(
            tmp_path,
            relative,
            """
class ModuleAdapter:
    def register(self):
        return self.shell._authorize_operation()
""".lstrip(),
        )

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("cross-component-private-access" in error for error in errors)

    def test_new_composition_late_binding_fails(self, tmp_path: Path) -> None:
        relative = "agent_libos/runtime/builder.py"
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)

        _write_source(
            tmp_path,
            relative,
            """
class RuntimeBuilder:
    def assemble(self, host):
        host.new_service.bind_runtime(host)
""".lstrip(),
        )

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("composition-late-binding" in error for error in errors)

    def test_builder_component_requires_runtime_declaration(self, tmp_path: Path) -> None:
        _write_source(
            tmp_path,
            "agent_libos/runtime/runtime.py",
            """
class Runtime:
    declared: object
""".lstrip(),
        )
        relative = "agent_libos/runtime/builder.py"
        _write_source(
            tmp_path,
            relative,
            """
class RuntimeBuilder:
    def assemble(self, host):
        host.declared = object()
""".lstrip(),
        )
        allowlist = _write_current_baseline(tmp_path)

        _write_source(
            tmp_path,
            relative,
            """
class RuntimeBuilder:
    def assemble(self, host):
        host.declared = object()
        host.implicit = object()
""".lstrip(),
        )

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("undeclared-runtime-component" in error for error in errors)

    def test_existing_component_coupling_passes_but_net_growth_fails(
        self, tmp_path: Path
    ) -> None:
        relative = "agent_libos/services/legacy.py"
        original = """
class LegacyService:
    def run(self):
        self.runtime.store.get_process("pid")
        self.checkpoint._insert_row("row")
""".lstrip()
        _write_source(tmp_path, relative, original)
        allowlist = _write_current_baseline(tmp_path)

        assert checker.check_architecture(tmp_path, allowlist) == []

        grown = original.replace(
            '        self.checkpoint._insert_row("row")\n',
            '        self.checkpoint._insert_row("row")\n'
            '        self.runtime.store.get_process("other")\n'
            '        self.checkpoint._insert_row("other")\n',
        )
        _write_source(tmp_path, relative, grown)

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("runtime-service-locator" in error for error in errors)
        assert any("cross-component-private-access" in error for error in errors)

    @pytest.mark.parametrize("runtime_name", ["runtime", "_runtime"])
    def test_runtime_alias_does_not_bypass_service_locator_guard(
        self,
        tmp_path: Path,
        runtime_name: str,
    ) -> None:
        relative = "agent_libos/services/alias.py"
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)

        _write_source(
            tmp_path,
            relative,
            f"""
class Service:
    def run(self):
        runtime = self.{runtime_name}
        return runtime.store.get_process("pid")
""".lstrip(),
        )

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("runtime-service-locator" in error for error in errors)

    def test_dynamic_runtime_alias_does_not_bypass_service_locator_guard(
        self,
        tmp_path: Path,
    ) -> None:
        relative = "agent_libos/services/dynamic_alias.py"
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)

        _write_source(
            tmp_path,
            relative,
            """
class Service:
    def run(self):
        runtime = getattr(self, "runtime", None)
        host = runtime
        return host.store.get_process("pid")
""".lstrip(),
        )

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("runtime-service-locator" in error for error in errors)

    def test_dynamic_runtime_service_access_fails(self, tmp_path: Path) -> None:
        relative = "agent_libos/services/dynamic_service.py"
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)

        _write_source(
            tmp_path,
            relative,
            """
class Service:
    def run(self):
        return getattr(getattr(self, "runtime", None), "store", None)
""".lstrip(),
        )

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("runtime-service-locator" in error for error in errors)

    def test_cross_component_private_field_access_fails(self, tmp_path: Path) -> None:
        relative = "agent_libos/services/private_state.py"
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)

        _write_source(
            tmp_path,
            relative,
            """
class Service:
    def run(self):
        return self.registry._entries
""".lstrip(),
        )

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("cross-component-private-access" in error for error in errors)

    def test_underscored_dependency_does_not_bypass_private_access_guard(
        self,
        tmp_path: Path,
    ) -> None:
        relative = "agent_libos/services/private_dependency.py"
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)

        _write_source(
            tmp_path,
            relative,
            """
class Service:
    def run(self):
        return self._registry._entries
""".lstrip(),
        )

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("cross-component-private-access" in error for error in errors)

    def test_local_alias_does_not_bypass_private_access_guard(
        self,
        tmp_path: Path,
    ) -> None:
        relative = "agent_libos/services/private_alias.py"
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)

        _write_source(
            tmp_path,
            relative,
            """
class Service:
    def run(self):
        registry = self._registry
        selected = registry
        return selected._entries
""".lstrip(),
        )

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("cross-component-private-access" in error for error in errors)

    def test_dynamic_private_access_does_not_bypass_guard(self, tmp_path: Path) -> None:
        relative = "agent_libos/services/dynamic_private.py"
        _write_source(tmp_path, relative, "VALUE = 1\n")
        allowlist = _write_current_baseline(tmp_path)

        _write_source(
            tmp_path,
            relative,
            """
class Service:
    def run(self):
        return getattr(self._registry, "_entries", None)
""".lstrip(),
        )

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("cross-component-private-access" in error for error in errors)

    def test_removed_component_debt_requires_budget_update(self, tmp_path: Path) -> None:
        relative = "agent_libos/services/private_dependency.py"
        _write_source(
            tmp_path,
            relative,
            """
class Service:
    def run(self):
        return self._registry._entries
""".lstrip(),
        )
        allowlist = _write_current_baseline(tmp_path)

        _write_source(tmp_path, relative, "VALUE = 1\n")

        errors = checker.check_architecture(tmp_path, allowlist)
        assert any("stale cross-component-private-access budget" in error for error in errors)

        _write_current_baseline(tmp_path)
        assert checker.check_architecture(tmp_path, allowlist) == []

    def test_component_ratchet_survives_method_renames(self, tmp_path: Path) -> None:
        relative = "agent_libos/services/legacy.py"
        original = """
class LegacyService:
    def old_name(self):
        self.runtime.store.get_process("pid")
        self.checkpoint._insert_row("row")
""".lstrip()
        _write_source(tmp_path, relative, original)
        allowlist = _write_current_baseline(tmp_path)

        _write_source(tmp_path, relative, original.replace("old_name", "new_name"))

        assert checker.check_architecture(tmp_path, allowlist) == []

    def test_direct_runtime_facade_call_is_not_a_service_locator(self, tmp_path: Path) -> None:
        _write_source(
            tmp_path,
            "agent_libos/api/sample.py",
            """
class Api:
    def open(self):
        return self.runtime.open()
""".lstrip(),
        )
        allowlist = _write_current_baseline(tmp_path)

        assert allowlist.is_file()
        assert checker.scan_architecture(tmp_path).violations == ()

    def test_existing_long_function_requires_budget_update_after_shrink(
        self,
        tmp_path: Path,
    ) -> None:
        relative = "agent_libos/services/long_function.py"
        _write_source(tmp_path, relative, _long_function(checker.MAX_FUNCTION_LINES))
        allowlist = _write_current_baseline(tmp_path)

        _write_source(
            tmp_path,
            relative,
            _long_function(checker.MAX_FUNCTION_LINES + 1),
        )
        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("long-function" in error for error in errors)

        _write_source(
            tmp_path,
            relative,
            _long_function(checker.MAX_FUNCTION_LINES - 1),
        )
        errors = checker.check_architecture(tmp_path, allowlist)
        assert any("stale long-function budget" in error for error in errors)

        _write_current_baseline(tmp_path)
        assert checker.check_architecture(tmp_path, allowlist) == []

    def test_new_long_function_fails_without_an_allowlist_entry(self, tmp_path: Path) -> None:
        _write_source(tmp_path, "agent_libos/__init__.py", "")
        allowlist = _write_current_baseline(tmp_path)

        _write_source(
            tmp_path,
            "agent_libos/services/new_long_function.py",
            _long_function(checker.MAX_FUNCTION_LINES),
        )

        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("long-function" in error for error in errors)

    def test_complexity_hotspot_requires_budget_update_after_shrink(
        self,
        tmp_path: Path,
    ) -> None:
        relative = "agent_libos/services/complexity.py"
        _write_source(
            tmp_path,
            relative,
            _complex_function(checker.COMPLEXITY_HOTSPOT_THRESHOLD),
        )
        allowlist = _write_current_baseline(tmp_path)

        _write_source(
            tmp_path,
            relative,
            _complex_function(checker.COMPLEXITY_HOTSPOT_THRESHOLD + 1),
        )
        errors = checker.check_architecture(tmp_path, allowlist)

        assert any("complexity-hotspot" in error for error in errors)

        _write_source(
            tmp_path,
            relative,
            _complex_function(checker.COMPLEXITY_HOTSPOT_THRESHOLD - 1),
        )
        errors = checker.check_architecture(tmp_path, allowlist)
        assert any("stale complexity-hotspot budget" in error for error in errors)

        _write_current_baseline(tmp_path)
        assert checker.check_architecture(tmp_path, allowlist) == []
