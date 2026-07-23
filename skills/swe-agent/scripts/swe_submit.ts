export async function run(args: Record<string, unknown>, libos: { syscall(name: string, args: unknown): Promise<any> }) {
  const payload = {
    status: String(args.status ?? "resolved"),
    summary: String(args.summary ?? ""),
    tests: Array.isArray(args.tests) ? args.tests.map((item) => String(item)) : [],
    residual_risks: Array.isArray(args.residual_risks) ? args.residual_risks.map((item) => String(item)) : [],
  };
  const exitResult = await libos.syscall("process.exit", { payload });
  return {
    submitted: true,
    payload,
    lifecycle: exitResult,
  };
}
