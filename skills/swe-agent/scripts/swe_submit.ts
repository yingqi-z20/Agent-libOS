export function run(
  args: Record<string, unknown>,
  libos: { syscall(name: string, args: unknown): Promise<any> },
) {
  void libos;
  const payload = {
    status: String(args.status ?? "resolved"),
    summary: String(args.summary ?? ""),
    tests: Array.isArray(args.tests) ? args.tests.map((item) => String(item)) : [],
    residual_risks: Array.isArray(args.residual_risks) ? args.residual_risks.map((item) => String(item)) : [],
  };
  return {
    prepared: true,
    payload,
    next_action: "process_exit",
    process_exit_args: { payload },
  };
}
