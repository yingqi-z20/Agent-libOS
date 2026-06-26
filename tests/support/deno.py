from __future__ import annotations

COUNT_CHARS_SOURCE = (
    'export function run(args, libos) { return { count: String(args.text || "").length }; }\n'
)

READ_FILE_SOURCE = (
    'export async function run(args, libos) { '
    'return await libos.syscall("filesystem.read_text", { path: args.path }); }\n'
)

WRITE_FILE_SOURCE = (
    'export async function run(args, libos) { '
    'return await libos.syscall("filesystem.write_text", '
    '{ path: args.path, content: args.content, overwrite: true }); }\n'
)

EXIT_AFTER_RESULT_SOURCE = (
    'export async function run(args, libos) { '
    'await libos.syscall("process.exit", { payload: { done: true } }); '
    'return { returned_after_exit_syscall: true }; }\n'
)

EXEC_AFTER_RESULT_SOURCE = (
    'export async function run(args, libos) { '
    'await libos.syscall("process.exec", '
    '{ image: "base-agent:v0", goal: "exec target", preserve_memory: true }); '
    'return { returned_after_exec_syscall: true }; }\n'
)

MISSING_EXEC_AFTER_RESULT_SOURCE = (
    'export async function run(args, libos) { '
    'await libos.syscall("process.exec", { image: "missing-image:v0", goal: "bad exec" }); '
    'return { returned: true }; }\n'
)

BAD_OUTPUT_SOURCE = (
    'export function run(args, libos) { return { count: "not-an-integer", extra: true }; }\n'
)
