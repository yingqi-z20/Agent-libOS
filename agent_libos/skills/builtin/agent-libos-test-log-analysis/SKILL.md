---
name: agent-libos-test-log-analysis
description: Extract a concise heuristic failure summary from already captured pytest output. Use when raw pytest logs are long and the FAILED, error, or AssertionError lines need triage; it does not execute tests.
allowed-tools: parse_pytest_log
---
# Analyze pytest logs

## Workflow

1. Supply the relevant raw pytest output text, including the failure summary and nearby assertions.
2. Use the extracted failed nodes, errors, and assertion lines to choose source or test files for inspection.
3. Return to the original non-truncated log whenever traceback context or final exit status matters.

## Boundaries and safety

- This parser is heuristic: it neither reads a file nor runs pytest.
- `failure_count` summarizes matched lines and is not an authoritative test outcome.
- Do not report success from an empty extraction if the supplied log was partial or truncated.

## Verify

Confirm important extracted nodes against the raw log and, after a fix, run the actual targeted test command separately.
