import { spawn } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";


type OfficialCheck = {
  id?: unknown;
  status?: unknown;
  specReferences?: unknown;
};

type OfficialScenario = {
  name: string;
  start(context: {
    specVersion: string;
    createServer: unknown;
  }): Promise<{
    serverUrl: string;
    authUrl?: string;
    context?: Record<string, unknown>;
  }>;
  stop(): Promise<void>;
  getChecks(): OfficialCheck[];
  authServer?: { getUrl(): string };
};

type ClientResult = {
  code: number;
  stdout: string;
  stderr: string;
  timedOut: boolean;
};


function requireLoopbackUrl(
  raw: string,
  options: { label: string; expectedPath: string },
): URL {
  const { label, expectedPath } = options;
  const selected = new URL(raw);
  if (
    selected.protocol !== "http:" ||
    selected.hostname !== "localhost" ||
    selected.port === "" ||
    selected.username !== "" ||
    selected.password !== "" ||
    selected.pathname !== expectedPath ||
    selected.search !== "" ||
    selected.hash !== ""
  ) {
    throw new Error(`official ${label} is not the reviewed loopback shape`);
  }
  return selected;
}


async function runClient(
  python: string,
  clientScript: string,
  serverUrl: string,
  scenario: string,
  context: Record<string, unknown>,
  timeoutMs: number,
): Promise<ClientResult> {
  return await new Promise((resolve, reject) => {
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    const child = spawn(python, [clientScript, "_client", serverUrl], {
      shell: false,
      stdio: ["ignore", "pipe", "pipe"],
      env: {
        ...process.env,
        MCP_CONFORMANCE_SCENARIO: scenario,
        MCP_CONFORMANCE_PROTOCOL_VERSION: "2026-07-28",
        MCP_CONFORMANCE_CONTEXT: JSON.stringify({
          name: scenario,
          ...context,
        }),
      },
    });
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdout = `${stdout}${chunk}`.slice(-16_384);
    });
    child.stderr.on("data", (chunk: string) => {
      stderr = `${stderr}${chunk}`.slice(-16_384);
    });
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
    }, timeoutMs);
    child.once("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.once("close", (code) => {
      clearTimeout(timer);
      resolve({
        code: code ?? -1,
        stdout,
        stderr,
        timedOut,
      });
    });
  });
}


async function main(): Promise<void> {
  const [checkoutRaw, scenarioName, python, clientScript, timeoutRaw, outputRaw] =
    process.argv.slice(2);
  if (
    !checkoutRaw ||
    !["auth/pre-registration", "auth/basic-cimd"].includes(scenarioName) ||
    !python ||
    !clientScript ||
    !timeoutRaw ||
    !outputRaw
  ) {
    throw new Error("invalid fixed-upstream OAuth harness arguments");
  }
  const timeoutMs = Number(timeoutRaw);
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 120000) {
    throw new Error("invalid fixed-upstream OAuth harness timeout");
  }
  const checkout = path.resolve(checkoutRaw);
  const scenarioModule = await import(
    pathToFileURL(path.join(checkout, "src", "scenarios", "index.ts")).href
  );
  const mockServerModule = await import(
    pathToFileURL(path.join(checkout, "src", "mock-server", "index.ts")).href
  );
  const scenario = scenarioModule.getScenario(scenarioName) as
    | OfficialScenario
    | undefined;
  if (!scenario || typeof scenario.start !== "function") {
    throw new Error("pinned official OAuth scenario is unavailable");
  }
  const output = path.resolve(outputRaw);
  await mkdir(output, { recursive: false });
  let client: ClientResult | undefined;
  let checks: OfficialCheck[] = [];
  try {
    const urls = await scenario.start({
      specVersion: "2026-07-28",
      createServer: mockServerModule.createServerFor("2026-07-28"),
    });
    const resource = requireLoopbackUrl(urls.serverUrl, {
      label: "Resource Server URL",
      expectedPath: "/mcp",
    });
    const issuerRaw = urls.authUrl ?? scenario.authServer?.getUrl();
    if (!issuerRaw) {
      throw new Error("pinned official scenario did not expose its fixture issuer");
    }
    const issuer = requireLoopbackUrl(issuerRaw, {
      label: "Authorization Server issuer",
      expectedPath: "/",
    });
    if (resource.origin === issuer.origin) {
      throw new Error("official OAuth fixture did not separate resource and issuer origins");
    }
    const upstreamContext = urls.context ?? {};
    let clientId: string;
    let clientSecret: string | null;
    let registrationMode: "preregistered" | "cimd";
    let tokenEndpointAuthMethod: "client_secret_basic" | "none";
    if (scenarioName === "auth/pre-registration") {
      if (
        typeof upstreamContext.client_id !== "string" ||
        typeof upstreamContext.client_secret !== "string"
      ) {
        throw new Error("pinned pre-registration context is incomplete");
      }
      clientId = upstreamContext.client_id;
      clientSecret = upstreamContext.client_secret;
      registrationMode = "preregistered";
      tokenEndpointAuthMethod = "client_secret_basic";
    } else {
      const cimdModule = await import(
        pathToFileURL(
          path.join(
            checkout,
            "src",
            "scenarios",
            "client",
            "auth",
            "basic-cimd.ts",
          ),
        ).href
      );
      if (typeof cimdModule.CIMD_CLIENT_METADATA_URL !== "string") {
        throw new Error("pinned CIMD client metadata URL is unavailable");
      }
      clientId = cimdModule.CIMD_CLIENT_METADATA_URL;
      clientSecret = null;
      registrationMode = "cimd";
      tokenEndpointAuthMethod = "none";
    }
    client = await runClient(
      python,
      clientScript,
      resource.href,
      scenarioName,
      {
        client_id: clientId,
        client_secret: clientSecret,
        registration_mode: registrationMode,
        token_endpoint_auth_method: tokenEndpointAuthMethod,
        trusted_resource_url: resource.href,
        trusted_issuer: issuer.origin,
        trusted_prm_url: new URL(
          "/.well-known/oauth-protected-resource/mcp",
          resource.origin,
        ).href,
        trusted_as_metadata_url: new URL(
          "/.well-known/oauth-authorization-server",
          issuer.origin,
        ).href,
      },
      timeoutMs,
    );
    checks = scenario.getChecks();
  } finally {
    await scenario.stop();
  }
  if (!client) {
    throw new Error("fixed-upstream OAuth client was not started");
  }
  if (client.timedOut) {
    throw new Error("fixed-upstream OAuth client timed out");
  }
  if (client.code !== 0) {
    throw new Error(`fixed-upstream OAuth client exited ${client.code}; output omitted`);
  }
  const failures = checks.filter(
    (check) => check.status === "FAILURE" || check.status === "WARNING",
  );
  if (failures.length > 0) {
    throw new Error(
      `fixed-upstream OAuth checks failed: ${failures
        .map((check) => `${String(check.id)}=${String(check.status)}`)
        .join(", ")}`,
    );
  }
  // Upstream details contain raw authorization query/body fields.  Emit only
  // the identity/status/spec projection after every failure check has passed;
  // the Python gate independently bounds and canonicalizes this evidence.
  const durableChecks = checks.map((check) => ({
    id: check.id,
    status: check.status,
    ...(check.specReferences === undefined
      ? {}
      : { specReferences: check.specReferences }),
  }));
  await writeFile(
    path.join(output, "checks.json"),
    `${JSON.stringify(durableChecks, null, 2)}\n`,
    "utf8",
  );
}


await main();
