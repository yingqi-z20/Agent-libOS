import { defineConfig, devices } from "@playwright/test";

const baseURL = process.env.AGENT_LIBOS_E2E_BASE_URL;
if (!baseURL) throw new Error("AGENT_LIBOS_E2E_BASE_URL is required");
assertLoopbackHttpUrl(baseURL, "AGENT_LIBOS_E2E_BASE_URL");

export default defineConfig({
  testDir: "./e2e",
  outputDir: ".playwright-results",
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  forbidOnly: Boolean(process.env.CI),
  reporter: process.env.CI ? [["line"], ["html", { open: "never" }]] : "line",
  timeout: 45_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL,
    serviceWorkers: "block",
    locale: "zh-CN",
    timezoneId: "Asia/Shanghai",
    colorScheme: "light",
    contextOptions: { reducedMotion: "reduce" },
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure"
  },
  projects: [
    {
      name: "desktop-chromium",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 920 } }
    },
    {
      name: "mobile-chromium",
      use: { ...devices["Pixel 7"], viewport: { width: 390, height: 844 } }
    }
  ]
});

function assertLoopbackHttpUrl(value: string, name: string): void {
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
}
