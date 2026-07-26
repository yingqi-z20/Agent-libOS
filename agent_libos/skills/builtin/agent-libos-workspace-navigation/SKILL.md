---
name: agent-libos-workspace-navigation
description: Read and inspect cwd-relative workspace directories and bounded text files, or select this AgentProcess working directory. Use to establish paths and textual evidence before acting; not for writes, binary or range reads, Object transfer, or Git metadata under .git.
allowed-tools: get_working_directory set_working_directory read_directory read_text_file
---
# Navigate the workspace

Filesystem tool inputs and outputs use different coordinates. Inputs must be relative to this process's current working directory (cwd); absolute paths are rejected even when they point inside the workspace, and unknown input fields are rejected. Returned `working_directory`, `path`, and directory-entry paths are canonical identities relative to the workspace root. Path separators follow the Host platform: on POSIX a backslash is an ordinary filename character and is preserved exactly. Preserve these distinctions throughout the task.

## Tool guide

### `get_working_directory`

- Input: an empty object.
- Output: `working_directory`, a workspace-root-relative directory such as `.` or `src/pkg`.
- Use it whenever the current base is not already established. It observes process state only; it does not list the directory or prove any file exists.

### `set_working_directory`

- Input: `path`, resolved relative to the old cwd. The provider's authorized canonical identity is stored exactly on success; it is not trimmed or re-normalized afterward.
- Output: the selected workspace-root-relative `working_directory`.
- The runtime first authorizes and observes an existing directory through filesystem-read authority, then records it as this process's cwd. Success proves that observation and the cwd update; it is not a transaction that freezes the directory for later calls.
- The new cwd affects later filesystem calls and shell subprocess cwd, and new child processes inherit it by default. It does not change the runtime host process cwd.

### `read_directory`

- Input: cwd-relative `path` and optional positive `limit` within the configured schema bound.
- Output: root-relative directory `path`, `entries`, returned-entry `count`, and `truncated`. Each entry always has `name`, root-relative `path`, `kind`, nullable `size_bytes`, and `modified_at`. With the default local provider, `kind` is `file`, `directory`, `symlink`, or `other`; `size_bytes` is an integer only for a regular file and is otherwise `null`.
- It lists one level only. `count` is the number returned, not the directory's total. There is no cursor or offset. If `truncated=true`, narrow to a known child or raise `limit` within the advertised bound; do not infer which omitted entries exist.

### `read_text_file`

- Input: cwd-relative `path`, requested `encoding`, and positive `max_bytes` within the configured hard bound.
- Output: root-relative `path`, decoded `content`, the `encoding` that successfully decoded it, raw-prefix `bytes_read`, `truncated`, and `content_sha256`. A complete read returns the SHA-256 of the complete raw file bytes; a truncated read returns `null`. Confirm the returned encoding matches the requested interpretation.
- Reading starts at byte zero and has no offset or range. `max_bytes` is a byte bound, not a character bound. When truncation cuts a multibyte character at the end, the incomplete character is omitted from `content`, while `bytes_read` still counts the selected raw byte; therefore encoded content length can be less than `bytes_read`. An invalid sequence elsewhere fails decoding instead of returning lossy text.

## Recommended workflow

1. Call `get_working_directory` unless the current cwd and its provenance are already clear.
2. Use `read_directory` on the narrowest useful directory. Establish each target's `kind` before treating it as a file or directory. Observing a `symlink` entry does not make it traversable by the generic local filesystem tools.
3. If changing location simplifies several later calls, call `set_working_directory` once and verify its returned root-relative identity. Its input is interpreted from the old cwd.
4. After a cwd change, pass names or newly computed cwd-relative paths. For example, from cwd `pkg`, an output identity `pkg/module.py` is read with input `module.py`; feeding `pkg/module.py` back unchanged would address `pkg/pkg/module.py`.
5. Use `read_text_file` with an explicit encoding when it is not known to be the runtime default and with the smallest sufficient byte budget. Check `truncated` before relying on endings, summaries, delimiters, or syntax closure.
6. For a truncated listing, subdivide by directory. For a large text file, first narrow the target or use a separately activated bounded extractor. Use Object-file transfer when content need not enter model context; do not repeatedly request a whole large file.

## Failure and recovery

- Filesystem reads and cwd selection require matching directory/file read authority and may require approval. A denial before the state probe intentionally reveals neither existence nor type; do not reinterpret it as "missing."
- Paths must remain inside the runtime workspace. The default local provider rejects every symlink or junction component after authorization, even when its resolved target remains inside the workspace; listing can therefore reveal a `symlink` that later read or cwd selection refuses to follow. It also rejects reads of regular files with multiple hard-link names. Do not treat either denial as evidence that the lexical path was missing.
- Any path component named `.git` is protected from these generic tools. Activate the appropriate Git Skill rather than attempting an alternate spelling, cwd, symlink, shell wrapper, or traversal.
- A decode error means the requested encoding did not produce valid text for the observed prefix. Recheck known file format and encoding; do not blindly cycle encodings and mistake plausible mojibake for evidence.
- Schema bounds are not persistence guarantees. Oversized arguments fail before dispatch, and a large normalized read result can fail without returned content. Narrow the read or use Object transfer; do not infer an empty file.
- A successful state-dependent read is a snapshot, not a lock. If another action may have changed the tree, observe it again. Finite read authority can be consumed after an observation, including an ambiguous provider failure.
- These tools are not binary-safe transfer or random-access APIs. Do not use decoded text, replacement characters, or a truncated prefix as a checksum or byte-for-byte assertion.

## Completion evidence

Before concluding navigation or handing a path to another Skill, record:

- the process cwd and whether each input was cwd-relative or each output was root-relative;
- the target's observed `kind` and canonical root-relative identity;
- for a listing, `count`, `limit`, and `truncated=false`, or an explicit statement that omitted entries remain unknown;
- for text, the returned successful encoding, `bytes_read`, `max_bytes`, and `truncated=false`, or a conclusion deliberately limited to the observed prefix;
- any later mutation or cwd change that invalidates earlier observations.

Stop when the exact target and sufficient bounded evidence are established. Use `agent-libos-workspace-editing` for mutations and the owning domain Skill for Git, Object, or binary work.
