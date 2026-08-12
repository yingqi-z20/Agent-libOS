from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class SyscallDescriptor:
    """Explicit route for one built-in syscall and its stable aliases."""

    name: str
    handler: str
    aliases: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


FILESYSTEM_SYSCALLS = (
    SyscallDescriptor("filesystem.read_text", "_filesystem_read_text", ("filesystem.read_text_file",)),
    SyscallDescriptor("filesystem.write_text", "_filesystem_write_text", ("filesystem.write_text_file",)),
    SyscallDescriptor("filesystem.read_directory", "_filesystem_read_directory", ("filesystem.list_directory",)),
    SyscallDescriptor("filesystem.write_directory", "_filesystem_write_directory", ("filesystem.make_directory",)),
    SyscallDescriptor("filesystem.delete_file", "_filesystem_delete_file"),
    SyscallDescriptor("filesystem.delete_directory", "_filesystem_delete_directory"),
)

MEMORY_SYSCALLS = (
    SyscallDescriptor("memory.create_namespace", "_memory_create_namespace"),
    SyscallDescriptor("memory.list_namespace", "_memory_list_namespace"),
    SyscallDescriptor("memory.create_object", "_memory_create_object"),
    SyscallDescriptor("memory.read_object", "_memory_read_object", ("memory.get_object",)),
    SyscallDescriptor("memory.append_object", "_memory_append_object", ("memory.append_memory_object",)),
)

HUMAN_SYSCALLS = (
    SyscallDescriptor("human.output", "_human_output", ("human_output",)),
    SyscallDescriptor("human.ask", "_human_ask", ("ask_human",)),
    SyscallDescriptor(
        "human.request_permission",
        "_request_permission",
        ("permission.request", "request_permission", "capability.request_permission"),
    ),
)

CAPABILITY_SYSCALLS = (
    SyscallDescriptor("capability.list", "_capability_list"),
    SyscallDescriptor("capability.inspect", "_capability_inspect"),
    SyscallDescriptor("capability.delegate", "_capability_delegate"),
    SyscallDescriptor("capability.revoke", "_capability_revoke"),
)

CLOCK_SYSCALLS = (
    SyscallDescriptor("clock.now", "_clock_now", ("time.now",)),
    SyscallDescriptor("clock.sleep", "_clock_sleep", ("sleep",)),
)

JSONRPC_SYSCALLS = (
    SyscallDescriptor("jsonrpc.list", "_jsonrpc_list"),
    SyscallDescriptor("jsonrpc.inspect", "_jsonrpc_inspect"),
    SyscallDescriptor("jsonrpc.call", "_jsonrpc_call"),
)

MCP_SYSCALLS = (
    SyscallDescriptor("mcp.list", "_mcp_list"),
    SyscallDescriptor("mcp.inspect", "_mcp_inspect"),
    SyscallDescriptor("mcp.tools", "_mcp_tools"),
    SyscallDescriptor("mcp.call", "_mcp_call"),
    SyscallDescriptor("mcp.resources", "_mcp_resources"),
    SyscallDescriptor("mcp.resource_read", "_mcp_resource_read"),
)

PROCESS_SYSCALLS = (
    SyscallDescriptor("process.cwd", "_process_cwd", ("process.get_working_directory",)),
    SyscallDescriptor("process.chdir", "_process_chdir", ("process.set_working_directory",)),
    SyscallDescriptor("process.fork", "_process_fork"),
    SyscallDescriptor("process.spawn_child", "_process_spawn_child"),
    SyscallDescriptor("process.wait", "_process_wait"),
    SyscallDescriptor("process.list_children", "_process_list_children"),
    SyscallDescriptor("process.signal", "_process_signal"),
    SyscallDescriptor("process.merge_child_memory", "_process_merge_child_memory"),
    SyscallDescriptor("process.send_message", "_process_send_message"),
    SyscallDescriptor("process.read_messages", "_process_read_messages_nonblocking"),
    SyscallDescriptor("process.receive_messages", "_process_receive_messages"),
    SyscallDescriptor("process.exec", "_process_exec"),
    SyscallDescriptor("process.exit", "_process_exit"),
)

CHECKPOINT_SYSCALLS = (
    SyscallDescriptor("checkpoint.create", "_checkpoint_create"),
    SyscallDescriptor("checkpoint.list", "_checkpoint_list"),
    SyscallDescriptor("checkpoint.inspect", "_checkpoint_inspect"),
    SyscallDescriptor("checkpoint.diff", "_checkpoint_diff"),
    SyscallDescriptor("checkpoint.restore", "_checkpoint_restore"),
    SyscallDescriptor("checkpoint.fork", "_checkpoint_fork", ("checkpoint.fork_from_checkpoint",)),
    SyscallDescriptor("checkpoint.replay", "_checkpoint_replay", ("checkpoint.replay_to_event",)),
)

SKILL_SYSCALLS = (
    SyscallDescriptor("skill.discover", "_skill_discover"),
    SyscallDescriptor("skill.inspect", "_skill_inspect"),
    SyscallDescriptor("skill.register_path", "_skill_register_path"),
    SyscallDescriptor("skill.activate", "_skill_activate"),
    SyscallDescriptor("skill.unload", "_skill_unload"),
    SyscallDescriptor("skill.read_resource", "_skill_read_resource"),
)

SHELL_SYSCALLS = (
    SyscallDescriptor("shell.run", "_shell_run", ("shell.run_command",)),
)

IMAGE_SYSCALLS = (
    SyscallDescriptor("image.list", "_image_list"),
    SyscallDescriptor("image.inspect", "_image_inspect"),
    SyscallDescriptor("image.commit_checkpoint", "_image_commit_checkpoint"),
    SyscallDescriptor("image.load_package", "_image_load_package"),
)

BUILTIN_SYSCALL_DESCRIPTORS = (
    *FILESYSTEM_SYSCALLS,
    *MEMORY_SYSCALLS,
    *HUMAN_SYSCALLS,
    *CAPABILITY_SYSCALLS,
    *CLOCK_SYSCALLS,
    *JSONRPC_SYSCALLS,
    *MCP_SYSCALLS,
    *PROCESS_SYSCALLS,
    *CHECKPOINT_SYSCALLS,
    *SKILL_SYSCALLS,
    *SHELL_SYSCALLS,
    *IMAGE_SYSCALLS,
)


def _index_descriptors(
    descriptors: tuple[SyscallDescriptor, ...],
) -> Mapping[str, SyscallDescriptor]:
    indexed: dict[str, SyscallDescriptor] = {}
    for descriptor in descriptors:
        for name in descriptor.names:
            normalized = name.strip()
            if not normalized or normalized != name:
                raise ValueError(f"invalid built-in syscall name: {name!r}")
            if normalized in indexed:
                raise ValueError(f"duplicate built-in syscall name: {normalized}")
            indexed[normalized] = descriptor
    return MappingProxyType(indexed)


BUILTIN_SYSCALL_ROUTES = _index_descriptors(BUILTIN_SYSCALL_DESCRIPTORS)
BUILTIN_SYSCALL_NAMES = frozenset(BUILTIN_SYSCALL_ROUTES)


__all__ = [
    "BUILTIN_SYSCALL_DESCRIPTORS",
    "BUILTIN_SYSCALL_NAMES",
    "BUILTIN_SYSCALL_ROUTES",
    "SyscallDescriptor",
]
