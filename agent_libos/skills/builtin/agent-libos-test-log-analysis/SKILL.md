---
name: agent-libos-test-log-analysis
description: Extract heuristic FAILED, error, and AssertionError lines from supplied pytest text. Use to triage an already captured bounded log; it does not read files, retain traceback context, run tests, return line indexes, or prove pass/fail.
allowed-tools: parse_pytest_log
---
# Triage pytest text

This parser is a deterministic marker extractor, not a pytest result parser. Keep the original exit code, raw log, traceback, and terminal summary as the authoritative evidence.

## Tool guide

### `parse_pytest_log`

- Input: one required string field, `log`. Unknown fields are rejected. It accepts captured text, not a filename, Object id, argv, or test selector.
- Output: string lists `failed`, `errors`, and `assertions`, plus integer `failure_count`. `failed` and `assertions` contain fully stripped lines. An `errors` item is the stripped source line with exactly its first two characters removed, so it can retain leading whitespace. These strings are not line numbers or indexes and contain no source offsets or neighboring context.
- Every input line is processed as `line.strip()`, then the first matching rule wins:
  1. a line beginning exactly with case-sensitive `FAILED ` is appended in full to `failed`;
  2. otherwise, a line matching case-sensitive `^E\s+` is appended to `errors` after exactly its `E` and the first following whitespace character are removed; any additional whitespace remains;
  3. otherwise, a line containing case-sensitive `AssertionError` is appended in full to `assertions`.
- `ERROR tests/...` does not match the `E` rule. A `FAILED ... AssertionError` line belongs only to `failed`; an `E   AssertionError` line belongs only to `errors`. Duplicate matching lines remain duplicates.
- `failure_count` is `len(failed) or len(assertions) or len(errors)`: the size of the first nonempty bucket in that exact order, not the sum and not necessarily pytest's failure total.

## Recommended workflow

1. Preserve the test command, process return code, full raw output location, and whether capture itself was truncated.
2. Supply a complete, bounded, relevant excerpt to `parse_pytest_log`. Include node-id summary lines and the terminal summary when they fit, but do not duplicate a huge log into a tool argument. Global tool-argument and result-persistence limits still apply.
3. Treat returned strings as search anchors. Find the matching occurrences in the original log and recover surrounding traceback, setup/teardown phase, captured output, and final summary yourself.
4. Deduplicate by node id and phase only after inspecting context. One test can emit multiple matching lines, and identical messages can occur in different tests.
5. Use the real test runner or its saved exit status to establish outcome. After a fix, run the narrowest affected test, then any required broader lane; analyze the new output rather than reusing stale markers.

When the original output is too large, first retain it as an authorized file or Object and extract bounded regions with an appropriate tool. `parse_pytest_log` itself has no cursor, offset, or paging contract, so repeatedly passing the same oversized log cannot recover omitted input or results. Passing only a head or tail is acceptable for local triage if explicitly labeled partial; it is not complete failure enumeration.

## Failure and recovery

- Zero matches means only that none of the three lexical patterns appeared. It does not mean tests passed, collection succeeded, pytest ran, or the input was complete.
- A positive `failure_count` is not a total. Always inspect all three lists and the raw terminal summary.
- The parser does not recognize every pytest failure form, including many collection/internal errors, short summaries with different prefixes, crashes, timeouts, killed processes, warnings promoted elsewhere, or plugin-specific output.
- Because the source line is initially stripped and error lines then lose exactly two characters while possibly retaining more leading whitespace, returned strings are not byte-for-byte log slices. Do not use them for offsets, signatures, or exact transcript reconstruction.
- Invalid or oversized tool arguments fail before parsing; an oversized normalized result can also fail persistence. Recover by selecting a smaller relevant excerpt, not by dropping the exit code or terminal summary that defines outcome.
- Partial input, truncated capture, missing stderr, or interleaved parallel logs must be reported as incomplete evidence. Do not fabricate node associations from ordering alone.

## Completion evidence

A sound triage report cites:

- the original test command and return code;
- whether stdout/stderr and the supplied excerpt were complete;
- each relevant returned line together with its recovered raw-log context and node id or phase when available;
- all three bucket sizes, while labeling `failure_count` with its first-nonempty-bucket semantics;
- the actual pytest terminal summary or an explicit statement that it was unavailable;
- the targeted rerun result after any fix.

Stop when the parser has supplied useful anchors. Do not present its lists as indexes, its count as a total, or absence of markers as a passing test result.
