function numberValue(value: unknown, fallback: number, min: number, max: number): number {
  const parsed = Number(value);
  const selected = Number.isFinite(parsed) ? Math.trunc(parsed) : fallback;
  return Math.max(min, Math.min(max, selected));
}

function parseMatches(text: string, stdoutTruncated: boolean): {
  records: Array<{ file: string; rendered: string }>;
  framingIncomplete: boolean;
} {
  const records: Array<{ file: string; rendered: string }> = [];
  let cursor = 0;
  let framingIncomplete = false;
  while (cursor < text.length) {
    const separator = text.indexOf("\0", cursor);
    if (separator < 0) {
      framingIncomplete = true;
      break;
    }
    const file = text.slice(cursor, separator);
    const lineStart = separator + 1;
    const newline = text.indexOf("\n", lineStart);
    if (newline < 0 && stdoutTruncated) {
      framingIncomplete = true;
      break;
    }
    const end = newline < 0 ? text.length : newline;
    const match = text.slice(lineStart, end).replace(/\r$/, "");
    if (file.length === 0 || match.length === 0) {
      framingIncomplete = true;
      break;
    }
    records.push({ file, rendered: `${file}:${match}` });
    cursor = newline < 0 ? text.length : newline + 1;
  }
  return { records, framingIncomplete };
}

export async function run(args: Record<string, unknown>, libos: { syscall(name: string, args: unknown): Promise<any> }) {
  const pattern = String(args.pattern ?? "");
  if (!pattern) throw new Error("pattern is required");
  const path = String(args.path ?? ".");
  const maxResults = numberValue(args.max_results, 50, 1, 200);
  const requestedTimeout = Number(args.timeout_s ?? 10);
  const timeoutSeconds = Number.isFinite(requestedTimeout)
    ? Math.max(Number.EPSILON, Math.min(10, requestedTimeout))
    : 10;
  const argv = [
    "rg",
    "-n",
    "--null",
    "--with-filename",
    "--hidden",
    "--glob",
    "!.git/*",
  ];
  if (args.literal !== false) argv.push("-F");
  argv.push("--", pattern, path);
  const result = await libos.syscall("shell.run", {
    argv,
    timeout_s: timeoutSeconds,
  });
  const text = String(result.stdout ?? "");
  const stderr = String(result.stderr ?? "");
  const stdoutTruncated = Boolean(result.stdout_truncated);
  const stderrTruncated = Boolean(result.stderr_truncated);
  const parsed = parseMatches(text, stdoutTruncated);
  const matchesIncomplete = stdoutTruncated || parsed.framingIncomplete;
  const outputIncomplete = matchesIncomplete || stderrTruncated;
  const selected = parsed.records.slice(0, maxResults);
  const matches = selected.map((record) => record.rendered);
  const files: string[] = [];
  const seen = new Set<string>();
  for (const record of selected) {
    if (!seen.has(record.file)) {
      seen.add(record.file);
      files.push(record.file);
    }
  }
  const observedOmittedMatches = Math.max(parsed.records.length - matches.length, 0);
  const emptyMessage =
    result.returncode === 0 &&
      text.length === 0 &&
      stderr.length === 0 &&
      !outputIncomplete
      ? "Your command ran successfully and did not produce any output."
      : "";
  return {
    argv: result.argv ?? argv,
    returncode: result.returncode,
    files,
    matches,
    omitted_matches: matchesIncomplete ? null : observedOmittedMatches,
    observed_omitted_matches: observedOmittedMatches,
    matches_incomplete: matchesIncomplete,
    stdout_truncated: stdoutTruncated,
    stderr_truncated: stderrTruncated,
    output_incomplete: outputIncomplete,
    stderr,
    message: emptyMessage,
  };
}
