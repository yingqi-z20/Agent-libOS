import AxeBuilder from "@axe-core/playwright";
import { readFileSync } from "node:fs";
import { expect, test, type BrowserContext, type Page } from "@playwright/test";

const fixture = {
  baseUrl: requiredUrl("AGENT_LIBOS_E2E_BASE_URL"),
  guiUrl: requiredUrl("AGENT_LIBOS_E2E_GUI_URL"),
  controlUrl: requiredUrl("AGENT_LIBOS_E2E_CONTROL_URL"),
  controlTokenFile: requiredEnv("AGENT_LIBOS_E2E_CONTROL_TOKEN_FILE"),
  liveCallId: requiredId("AGENT_LIBOS_E2E_LIVE_CALL_ID"),
  summaryCallId: requiredId("AGENT_LIBOS_E2E_SUMMARY_CALL_ID"),
  hashCallId: requiredId("AGENT_LIBOS_E2E_HASH_CALL_ID"),
  conflictCallId: requiredId("AGENT_LIBOS_E2E_CONFLICT_CALL_ID"),
  limitedCallId: requiredId("AGENT_LIBOS_E2E_LIMITED_CALL_ID"),
  oldestCallId: requiredId("AGENT_LIBOS_E2E_OLDEST_CALL_ID"),
  expectedCallCount: Number(requiredEnv("AGENT_LIBOS_E2E_EXPECTED_CALL_COUNT"))
};

if (!Number.isSafeInteger(fixture.expectedCallCount) || fixture.expectedCallCount <= 50) {
  throw new Error("AGENT_LIBOS_E2E_EXPECTED_CALL_COUNT must exceed one 50-call page");
}

type PageAudit = {
  blockedHttp: string[];
  blockedWebSockets: string[];
  consoleMessages: string[];
  pageErrors: string[];
  contentRequests: Array<{ field: string; attempt: string }>;
};

const audits = new WeakMap<Page, PageAudit>();
const allowedHttpOrigins = new Set([
  new URL(fixture.baseUrl).origin,
  new URL(fixture.guiUrl).origin
]);
const allowedWebSocketOrigins = new Set(
  [new URL(fixture.baseUrl).origin.replace(/^http/, "ws")]
);

test.beforeEach(async ({ context, page }) => {
  const audit: PageAudit = {
    blockedHttp: [],
    blockedWebSockets: [],
    consoleMessages: [],
    pageErrors: [],
    contentRequests: []
  };
  audits.set(page, audit);
  page.on("console", (message) => audit.consoleMessages.push(message.text()));
  page.on("pageerror", (error) => audit.pageErrors.push(error.message));
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.pathname.endsWith("/content")) {
      audit.contentRequests.push({
        field: url.searchParams.get("field") ?? "",
        attempt: url.searchParams.get("attempt_sequence") ?? ""
      });
    }
  });

  await installNetworkBoundary(context, audit);
});

test("operator jumps from the timeline into a real retried Provider trace", async ({ page }) => {
  await useView(page, "operator");
  await page.goto("/");

  const jump = page.getByRole("button", { name: "查看 Provider 轨迹", exact: true });
  await expect(jump).toHaveCount(1);
  await jump.focus();
  await jump.press("Enter");

  const panel = page.getByTestId("provider-trace-panel");
  await expect(panel).toBeVisible();
  await expect(panel).toBeFocused();
  await expect(
    panel.getByTestId(`provider-trace-call-${fixture.liveCallId}`)
  ).toHaveAttribute("aria-pressed", "true");

  const firstAttempt = panel.getByTestId("provider-attempt-1");
  const fallbackAttempt = panel.getByTestId("provider-attempt-2");
  const selectedAttempt = panel.getByTestId("provider-attempt-3");
  await expect(firstAttempt).toContainText("initial");
  await expect(firstAttempt).toContainText("error");
  await expect(fallbackAttempt).toContainText("responses_to_chat");
  await expect(fallbackAttempt).toContainText("error");
  await expect(selectedAttempt).toContainText("transport_retry");
  await expect(selectedAttempt).toContainText("ok");
  expect(auditFor(page).contentRequests).toEqual([]);

  const reasoning = selectedAttempt.getByRole("region", { name: "返回的推理" });
  await reasoning.getByRole("button", { name: "显示内容", exact: true }).click();
  await expect(reasoning).toContainText("E2E_REASONING_START");
  expect(auditFor(page).contentRequests).toEqual([
    { field: "attempt_reasoning", attempt: "3" }
  ]);
  await loadAllContentChunks(page, reasoning);
  await expect(reasoning).toContainText("E2E_REASONING_END");
  await expect(reasoning).toContainText("<script>window.__providerTracePwned=true</script>");
  await expect(reasoning.getByRole("link")).toHaveCount(0);
  await expect.poll(() => page.evaluate(
    () => (window as Window & { __providerTracePwned?: boolean }).__providerTracePwned
  )).toBeUndefined();

  const output = selectedAttempt.getByRole("region", { name: "尝试输出" });
  await output.getByRole("button", { name: "显示内容", exact: true }).click();
  await expect(output).toContainText("E2E_OUTPUT_COMPLETE");

  const tools = selectedAttempt.getByRole("region", { name: "尝试工具动作" });
  await tools.getByRole("button", { name: "显示内容", exact: true }).click();
  await expect(tools).toContainText("process_exit");

  await panel.getByText("请求与底层响应字段", { exact: true }).click();
  const requestOptions = panel.getByRole("region", { name: "请求选项" });
  await requestOptions.getByRole("button", { name: "显示内容", exact: true }).click();
  await expect(requestOptions).toContainText("fallback_json_actions_enabled");
  await expect(requestOptions).toContainText("fallback_json_action_used");
  await expect(requestOptions.locator("pre")).toContainText(/fallback_json_actions_enabled[\s\S]*true/);
  await expect(requestOptions.locator("pre")).toContainText(/fallback_json_action_used[\s\S]*false/);

  await expectNoSeriousAxeViolations(page, "[data-testid='provider-trace-panel']");
  await expectNoCredentialPersistence(page);
});

test("user reasoning drawer is generic, keyboard-bounded, and session-only", async ({ page }) => {
  await useView(page, "user");
  await page.goto("/");

  const toggle = page.getByTestId("user-reasoning-toggle");
  await expect(toggle).toHaveAttribute("aria-pressed", "false");
  await toggle.focus();
  await toggle.press("Enter");

  const drawer = page.getByTestId("user-reasoning-drawer");
  await expect(drawer).toBeVisible();
  await expect(page.getByTestId("close-reasoning-drawer")).toBeFocused();
  const genericCall = drawer.getByTestId("provider-trace-call").first();
  await expect(genericCall).toHaveAttribute("aria-pressed", "true");
  await expect(drawer).not.toContainText(fixture.liveCallId);
  await expect(drawer.getByText("请求与底层响应字段")).toHaveCount(0);
  expect(auditFor(page).contentRequests).toEqual([]);

  const reasoning = drawer.getByRole("region", { name: "返回的推理" });
  await reasoning.getByRole("button", { name: "显示内容", exact: true }).click();
  await expect(reasoning).toContainText("E2E_REASONING_START");
  expect(auditFor(page).contentRequests).toEqual([
    { field: "attempt_reasoning", attempt: "3" }
  ]);
  await expectNoSeriousAxeViolations(page, "[data-testid='user-reasoning-drawer']");

  const close = page.getByTestId("close-reasoning-drawer");
  await close.focus();
  await close.press("Shift+Tab");
  await expect.poll(() => page.evaluate(() => {
    const modal = document.querySelector("[data-testid='user-reasoning-drawer']");
    return Boolean(modal && document.activeElement && modal.contains(document.activeElement));
  })).toBe(true);
  await page.keyboard.press("Escape");
  await expect(drawer).toHaveCount(0);
  await expect(toggle).toBeFocused();

  await toggle.press("Enter");
  await expect(page.getByTestId("user-reasoning-drawer")).toBeVisible();
  await page.reload();
  await expect(page.getByTestId("user-reasoning-toggle")).toHaveAttribute("aria-pressed", "false");
  await expect(page.getByTestId("user-reasoning-drawer")).toHaveCount(0);
  await expectNoCredentialPersistence(page);
});

test("operator paginates calls and clears stale chunks after a retention race", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chromium", "desktop coverage is sufficient for data-window semantics");
  await useView(page, "operator");
  await page.goto("/");
  await page.getByRole("combobox", { name: "进程详情分区", exact: true }).selectOption("llmCalls");

  const panel = page.getByTestId("provider-trace-panel");
  const callButtons = panel.locator('[data-testid^="provider-trace-call-"]');
  await expect(callButtons).toHaveCount(50);
  const loadEarlier = panel.getByRole("button", { name: "加载更早调用", exact: true });
  await loadEarlier.focus();
  await loadEarlier.press("Enter");
  await expect(callButtons).toHaveCount(fixture.expectedCallCount);
  await expect(panel.getByTestId(`provider-trace-call-${fixture.oldestCallId}`)).toBeVisible();

  await panel.getByTestId(`provider-trace-call-${fixture.summaryCallId}`).click();
  await expect(panel.getByText("summary", { exact: true })).toBeVisible();
  await expect(panel.getByText("该调用没有保留尝试详情。", { exact: true })).toBeVisible();
  await panel.getByText("请求与底层响应字段", { exact: true }).click();
  await expect(panel.getByRole("region", { name: "输入消息" }).getByRole("button", { name: "显示内容", exact: true })).toHaveCount(0);
  await expect(panel.getByRole("region", { name: "工具定义" }).getByRole("button", { name: "显示内容", exact: true })).toHaveCount(0);
  await expect(panel.getByRole("region", { name: "有界原始响应" }).getByRole("button", { name: "显示内容", exact: true })).toHaveCount(0);
  await expect(panel.getByRole("region", { name: "最终响应" }).getByRole("button", { name: "显示内容", exact: true })).toHaveCount(0);
  await expect(panel.getByRole("region", { name: "请求选项" }).getByRole("button", { name: "显示内容", exact: true })).toHaveCount(1);

  await panel.getByTestId(`provider-trace-call-${fixture.hashCallId}`).click();
  await expect(panel.getByText("hash_only", { exact: true })).toBeVisible();
  await expect(panel.getByText("该调用没有保留尝试详情。", { exact: true })).toBeVisible();

  await panel.getByTestId(`provider-trace-call-${fixture.limitedCallId}`).click();
  const limitedAttempt = panel.getByTestId("provider-attempt-1");
  await limitedAttempt.getByText("推理块元数据", { exact: true }).click();
  await expect(limitedAttempt).toContainText("omitted");
  const limitedReasoning = limitedAttempt.getByRole("region", { name: "返回的推理" });
  await limitedReasoning.getByRole("button", { name: "显示内容", exact: true }).click();
  await expect(limitedReasoning).toContainText("E2E_LIMITED_VISIBLE");
  await expect(limitedReasoning).toContainText("Provider 轨迹已受边界限制；省略内容无法重建。");

  await panel.getByTestId(`provider-trace-call-${fixture.conflictCallId}`).click();
  await expect(panel.getByText("full", { exact: true })).toBeVisible();
  await panel.getByText("请求与底层响应字段", { exact: true }).click();
  const responseContent = panel.getByRole("region", { name: "最终响应" });
  await responseContent.getByRole("button", { name: "显示内容", exact: true }).click();
  await expect(responseContent).toContainText("E2E_CONFLICT_START");
  const nextChunk = responseContent.getByRole("button", { name: "加载下一块", exact: true });
  await expect(nextChunk).toBeVisible();

  await downgradeConflictRecord();
  const changedResponse = page.waitForResponse((response) => (
    response.status() === 409
    && response.url().includes(`/llm-calls/${fixture.conflictCallId}/content`)
  ));
  await nextChunk.click();
  await changedResponse;
  await expect(responseContent.getByRole("alert")).toHaveText(
    "该内容已变化或降低保留层级，请重新加载调用后继续。"
  );
  await expect(responseContent.locator("pre")).toHaveCount(0);

  await expectNoSeriousAxeViolations(page, "[data-testid='provider-trace-panel']");
  await expectNoCredentialPersistence(page);
});

test("browser traffic cannot leave the allowlisted loopback origins", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chromium", "one Chromium network-boundary check is sufficient");
  await page.goto("/");
  const result = await page.evaluate(async () => {
    let http = "unexpected-success";
    try {
      await fetch("https://nonloopback.invalid/e2e-http-block");
    } catch {
      http = "blocked";
    }
    const websocket = await new Promise<string>((resolve) => {
      const socket = new WebSocket("wss://nonloopback.invalid/e2e-ws-block");
      const timer = window.setTimeout(() => resolve("timeout"), 3_000);
      socket.addEventListener("open", () => {
        window.clearTimeout(timer);
        resolve("unexpected-success");
      }, { once: true });
      socket.addEventListener("error", () => {
        window.clearTimeout(timer);
        resolve("blocked");
      }, { once: true });
      socket.addEventListener("close", () => {
        window.clearTimeout(timer);
        resolve("blocked");
      }, { once: true });
    });
    return { http, websocket };
  });

  expect(result).toEqual({ http: "blocked", websocket: "blocked" });
  expect(auditFor(page).blockedHttp).toContain("https://nonloopback.invalid/e2e-http-block");
  expect(auditFor(page).blockedWebSockets).toContain("wss://nonloopback.invalid/e2e-ws-block");
  await expectNoCredentialPersistence(page);
});

async function installNetworkBoundary(context: BrowserContext, audit: PageAudit) {
  await context.route("**/*", async (route) => {
    const requestUrl = route.request().url();
    const url = new URL(requestUrl);
    if (["data:", "blob:", "about:"].includes(url.protocol)) {
      await route.continue();
      return;
    }
    if (["http:", "https:"].includes(url.protocol) && allowedHttpOrigins.has(url.origin)) {
      await route.continue();
      return;
    }
    audit.blockedHttp.push(requestUrl);
    await route.abort("blockedbyclient");
  });
  await context.routeWebSocket("**/*", async (route) => {
    const requestUrl = route.url();
    const url = new URL(requestUrl);
    if (["ws:", "wss:"].includes(url.protocol) && allowedWebSocketOrigins.has(url.origin)) {
      route.connectToServer();
      return;
    }
    audit.blockedWebSockets.push(requestUrl);
    await route.close({ code: 1008, reason: "E2E network boundary" });
  });
}

async function useView(page: Page, view: "operator" | "user") {
  await page.addInitScript((selectedView) => {
    localStorage.setItem("agent-libos.gui.view", selectedView);
    localStorage.setItem("agent-libos.gui.language", "zh-CN");
  }, view);
}

async function loadAllContentChunks(page: Page, region: ReturnType<Page["getByRole"]>) {
  const button = region.getByRole("button", { name: "加载下一块", exact: true });
  for (let index = 0; index < 16 && await button.count(); index += 1) {
    const previousLength = (await region.locator("pre").textContent())?.length ?? 0;
    const response = page.waitForResponse((candidate) => (
      candidate.status() === 200
      && candidate.url().includes("/llm-calls/")
      && candidate.url().includes("/content?")
      && candidate.url().includes("field=attempt_reasoning")
    ));
    await button.click();
    await response;
    await expect.poll(async () => (
      ((await region.locator("pre").textContent())?.length ?? 0) > previousLength
      && (await button.count() === 0 || await button.isEnabled())
    )).toBe(true);
  }
  await expect(button).toHaveCount(0);
}

async function downgradeConflictRecord() {
  const token = readFileSync(fixture.controlTokenFile, "utf8").trim();
  if (!token.startsWith("e2e_control_secret_")) {
    throw new Error("invalid E2E control token file");
  }
  const response = await fetch(fixture.controlUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-E2E-Control-Token": token
    },
    body: JSON.stringify({ call_id: fixture.conflictCallId, target: "summary" }),
    redirect: "error"
  });
  if (!response.ok) throw new Error(`fixture retention control failed (${response.status})`);
  const value = await response.json() as { ok?: boolean };
  if (value.ok !== true) throw new Error("fixture retention control rejected the transition");
}

async function expectNoSeriousAxeViolations(page: Page, selector: string) {
  await expect(page.locator(selector)).toBeVisible();
  const result = await new AxeBuilder({ page }).analyze();
  const violations = [];
  for (const item of result.violations) {
    if (item.impact !== "serious" && item.impact !== "critical") continue;
    const nodes = [];
    for (const node of item.nodes) {
      const inScope = await page.evaluate(({ scopeSelector, targets }) => {
        const scope = document.querySelector(scopeSelector);
        if (!scope) return false;
        for (const target of targets) {
          if (typeof target !== "string") continue;
          try {
            if ([...document.querySelectorAll(target)].some((element) => scope.contains(element))) {
              return true;
            }
          } catch {
            // Axe may emit a shadow/frame selector that is not a document CSS selector.
          }
        }
        return false;
      }, { scopeSelector: selector, targets: node.target });
      if (inScope) nodes.push(node);
    }
    if (nodes.length) violations.push({ ...item, nodes });
  }
  expect(violations, JSON.stringify(violations, null, 2)).toEqual([]);
}

async function expectNoCredentialPersistence(page: Page) {
  const state = await page.evaluate(() => ({
    url: location.href,
    html: document.documentElement.innerHTML,
    local: Object.entries(localStorage),
    session: Object.entries(sessionStorage)
  }));
  const cookies = await page.context().cookies();
  const audit = auditFor(page);
  const serialized = JSON.stringify({
    state,
    cookies,
    consoleMessages: audit.consoleMessages,
    pageErrors: audit.pageErrors
  });
  expect(state.url).not.toMatch(/[?&#](?:token|authorization|api_key)=/i);
  expect(serialized).not.toMatch(/e2e_(?:gui|provider|control)_secret_/i);
  expect(serialized).not.toMatch(/Bearer\s+[A-Za-z0-9._~-]+/i);
  expect(audit.pageErrors).toEqual([]);
}

function auditFor(page: Page): PageAudit {
  const audit = audits.get(page);
  if (!audit) throw new Error("page audit was not installed");
  return audit;
}

function requiredEnv(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is required`);
  return value;
}

function requiredUrl(name: string): string {
  const value = requiredEnv(name);
  const parsed = new URL(value);
  if (
    parsed.protocol !== "http:"
    || parsed.hostname !== "127.0.0.1"
    || !parsed.port
    || parsed.username
    || parsed.password
  ) {
    throw new Error(`${name} must be an explicit credential-free loopback HTTP URL`);
  }
  return value;
}

function requiredId(name: string): string {
  const value = requiredEnv(name);
  if (!/^[A-Za-z0-9_.:-]+$/.test(value)) throw new Error(`${name} is invalid`);
  return value;
}
