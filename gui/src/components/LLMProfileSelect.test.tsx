import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { LLMProfileSummary } from "../api/types";
import { I18nProvider } from "../i18n";
import { LLMProfileSelect } from "./LLMProfileSelect";

describe("LLMProfileSelect", () => {
  it("renders profile choices and warns when the selected env var is missing", () => {
    const html = renderToStaticMarkup(
      <I18nProvider>
        <LLMProfileSelect
          profiles={[profile("default", "config", false, true), profile("qwen3.7-max", "user", true, false)]}
          value="qwen3.7-max"
          onChange={() => undefined}
          onCreate={async () => true}
          onUpdate={async () => true}
          onDelete={async () => true}
        />
      </I18nProvider>
    );

    expect(html).toMatch(/Image\/runtime default|镜像\/运行时默认/);
    expect(html).toContain("qwen3.7-max");
    expect(html).toMatch(/Environment variable QWEN_API_KEY is not set|环境变量 QWEN_API_KEY 尚未设置/);
  });

  it("renders the manager with config profiles read-only", () => {
    const html = renderToStaticMarkup(
      <I18nProvider>
        <LLMProfileSelect
          profiles={[profile("default", "config", false, true), profile("kimi-k2.7-code", "user", true, true)]}
          value="default"
          initialManageOpen
          onChange={() => undefined}
          onCreate={async () => true}
          onUpdate={async () => true}
          onDelete={async () => true}
        />
      </I18nProvider>
    );

    expect(html).toMatch(/Model profiles|模型 Profiles/);
    expect(html).toMatch(/Config profiles are read-only|配置文件中的 profile 只读/);
    expect(html).toMatch(/Context window tokens|上下文窗口 tokens/);
    expect(html).toContain("kimi-k2.7-code");
    expect(html).toContain("disabled=\"\"");
  });
});

function profile(
  profileId: string,
  source: "config" | "user",
  editable: boolean,
  apiKeyEnvPresent: boolean
): LLMProfileSummary {
  return {
    profile_id: profileId,
    model: profileId === "default" ? "gpt-5.5" : profileId,
    base_url: source === "user" ? "https://example.test/v1" : null,
    api_key_env: profileId.startsWith("qwen") ? "QWEN_API_KEY" : "OPENAI_API_KEY",
    api_key_env_present: apiKeyEnvPresent,
    api_mode: "chat",
    timeout_s: null,
    max_retries: null,
    store: null,
    reasoning_effort: null,
    verbosity: null,
    parallel_tool_calls: null,
    auto_wait_on_empty_tool_calls: null,
    temperature: null,
    max_tokens: null,
    context_window_tokens: null,
    allow_custom_base_url: source === "user",
    source,
    editable,
    is_default: profileId === "default"
  };
}
