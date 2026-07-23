function numberValue(value: unknown, fallback: number, min: number, max: number): number {
  const parsed = Number(value);
  const selected = Number.isFinite(parsed) ? Math.trunc(parsed) : fallback;
  return Math.max(min, Math.min(max, selected));
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
  const argv = ["rg", "-n", "--hidden", "--glob", "!.git/*"];
  if (args.literal !== false) argv.push("-F");
  argv.push("--", pattern, path);
  const result = await libos.syscall("shell.run", {
    argv,
    timeout_s: timeoutSeconds,
  });
  const text = String(result.stdout ?? "");
  const rawMatches = text.split(/\r?\n/).filter((line) => line.length > 0);
  const matches = rawMatches.slice(0, maxResults);
  const files = [];
  const seen = new Set<string>();
  for (const line of rawMatches) {
    const index = line.indexOf(":");
    const file = index >= 0 ? line.slice(0, index) : line;
    if (!seen.has(file)) {
      seen.add(file);
      files.push(file);
    }
  }
  const emptyMessage = result.returncode === 0 && text.length === 0
    ? "Your command ran successfully and did not produce any output."
    : "";
  return {
    argv: result.argv ?? argv,
    returncode: result.returncode,
    files,
    matches,
    omitted_matches: Math.max(rawMatches.length - matches.length, 0),
    stderr: result.stderr ?? "",
    message: emptyMessage,
  };
}
