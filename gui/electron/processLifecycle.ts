export type ChildExitState = {
  exitCode: number | null;
  signalCode: NodeJS.Signals | null;
  /** Present on ChildProcess, deliberately ignored for liveness decisions. */
  killed?: boolean;
};

export type ServerConnection = { url: string; token: string; db: string };
export type StartupOutputState = { text: string; scanOffset: number };

export function isChildAlive(child: ChildExitState | null | undefined): boolean {
  return Boolean(child && child.exitCode === null && child.signalCode === null);
}

export function appendStartupOutput(
  current: string,
  chunk: Buffer,
  maxBytes: number,
  streamName = "output"
): string {
  if (!Number.isSafeInteger(maxBytes) || maxBytes <= 0) {
    throw new Error("GUI server startup output limit must be a positive safe integer.");
  }
  const currentBytes = Buffer.byteLength(current, "utf8");
  if (chunk.byteLength > maxBytes - currentBytes) {
    throw new Error(`GUI server startup ${streamName} exceeded ${maxBytes} bytes.`);
  }
  return Buffer.concat([Buffer.from(current, "utf8"), chunk]).toString("utf8");
}

export function consumeStartupOutput(
  current: StartupOutputState,
  chunk: Buffer,
  maxBytes: number
): { state: StartupOutputState; connection: ServerConnection | null; frame: string | null } {
  const text = appendStartupOutput(current.text, chunk, maxBytes, "stdout");
  let scanOffset = current.scanOffset;
  while (scanOffset < text.length) {
    const newline = text.indexOf("\n", scanOffset);
    if (newline < 0) break;
    const frame = text.slice(scanOffset, newline).replace(/\r$/, "");
    scanOffset = newline + 1;
    const selected = frame.trim();
    if (!selected.startsWith("{")) continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(selected);
    } catch (error) {
      throw new Error(`GUI server emitted an invalid complete startup JSON frame: ${String(error)}`);
    }
    const connection = serverConnectionFromFrame(parsed);
    if (connection) return { state: { text, scanOffset }, connection, frame: selected };
  }
  return { state: { text, scanOffset }, connection: null, frame: null };
}

export function serverConnectionFromFrame(value: unknown): ServerConnection | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  const connectionKeys = ["url", "token", "db"];
  if (!connectionKeys.some((key) => key in record)) return null;
  if (
    typeof record.url !== "string"
    || typeof record.token !== "string"
    || !record.token.trim()
    || typeof record.db !== "string"
    || !record.db.trim()
  ) {
    throw new Error("GUI server emitted a malformed startup connection frame.");
  }
  let url: URL;
  try {
    url = new URL(record.url);
  } catch {
    throw new Error("GUI server emitted an invalid startup URL.");
  }
  const loopback = url.hostname === "localhost" || url.hostname === "127.0.0.1" || url.hostname === "[::1]";
  if (
    url.protocol !== "http:"
    || !loopback
    || !url.port
    || url.pathname !== "/"
    || url.search
    || url.hash
    || url.username
    || url.password
  ) {
    throw new Error("GUI server startup URL must be an unauthenticated loopback HTTP origin.");
  }
  return { url: record.url, token: record.token, db: record.db };
}

export async function withStartupFailureCleanup<T>(
  operation: () => Promise<T>,
  cleanup: () => Promise<void>
): Promise<T> {
  try {
    return await operation();
  } catch (error) {
    await cleanup();
    throw error;
  }
}

export async function cleanupBeforeExit(cleanup: () => Promise<void>, exit: () => void): Promise<void> {
  try {
    await cleanup();
  } finally {
    exit();
  }
}
