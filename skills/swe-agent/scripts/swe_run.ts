export async function run(args: Record<string, unknown>, libos: { syscall(name: string, args: unknown): Promise<any> }) {
  if (!Array.isArray(args.argv) || args.argv.length === 0) {
    throw new Error("argv must be a non-empty string array");
  }
  const argv = args.argv.map((item) => String(item));
  const requestedTimeout = Number(args.timeout_s ?? 55);
  const timeoutSeconds = Number.isFinite(requestedTimeout)
    ? Math.max(Number.EPSILON, Math.min(55, requestedTimeout))
    : 55;
  const result = await libos.syscall("shell.run", {
    argv,
    timeout_s: timeoutSeconds,
  });
  const stdout = String(result.stdout ?? "");
  const stderr = String(result.stderr ?? "");
  const message = stdout.length === 0 && stderr.length === 0
    ? "Your command ran successfully and did not produce any output."
    : "";
  return {
    argv: result.argv ?? argv,
    returncode: result.returncode,
    stdout,
    stderr,
    stdout_truncated: Boolean(result.stdout_truncated),
    stderr_truncated: Boolean(result.stderr_truncated),
    message,
  };
}
