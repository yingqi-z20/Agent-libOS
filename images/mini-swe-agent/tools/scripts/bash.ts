const TIMEOUT_SECONDS = 30;
const OUTPUT_LIMIT = 10000;
const OUTPUT_EDGE = 5000;

type LibOS = {
  syscall(name: string, args?: Record<string, unknown>): Promise<any>;
};

function commandText(value: unknown): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error("command must be a non-empty string");
  }
  return value;
}

function observation(returncode: number, output: string, exceptionInfo = ""): Record<string, unknown> {
  if (output.length <= OUTPUT_LIMIT) {
    return {
      returncode,
      output,
      exception_info: exceptionInfo,
    };
  }
  return {
    returncode,
    output_head: output.slice(0, OUTPUT_EDGE),
    output_tail: output.slice(-OUTPUT_EDGE),
    elided_chars: output.length - OUTPUT_EDGE * 2,
    warning: `Output was longer than ${OUTPUT_LIMIT} characters and was truncated to head/tail windows.`,
    exception_info: exceptionInfo,
  };
}

export async function run(args: Record<string, unknown>, libos: LibOS): Promise<Record<string, unknown>> {
  const command = commandText(args.command);
  const submit = args.submit === true;
  try {
    const result = await libos.syscall("shell.run", {
      argv: ["bash", "-lc", `exec 2>&1; ${command}`],
      timeout_s: TIMEOUT_SECONDS,
    });
    const returncode = Number.isFinite(Number(result.returncode)) ? Math.trunc(Number(result.returncode)) : -1;
    const stdout = String(result.stdout ?? "");
    const stderr = String(result.stderr ?? "");
    const output = stdout + stderr;
    if (returncode === 0 && submit) {
      await libos.syscall("process.exit", {
        payload: {
          status: "submitted",
          output,
        },
      });
    }
    return observation(returncode, output);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return observation(-1, "", message);
  }
}
