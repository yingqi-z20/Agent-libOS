// @vitest-environment jsdom

import { act } from "react";
import { createRoot } from "react-dom/client";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { LLMProfileSummary } from "../api/types";
import { I18nProvider } from "../i18n";
import { LLMProfileSelect, parseProfileNumber } from "./LLMProfileSelect";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

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
    expect(html).toMatch(/Reasoning effort|推理强度/);
    expect(html).toMatch(/Prompt cache retention|Prompt 缓存保留/);
    expect(html).toContain('<option value="in_memory">in_memory</option>');
    expect(html).not.toContain('value="in-memory"');
    expect(html).toMatch(/Reuse Responses chain|复用 Responses 调用链/);
    expect(html).toContain("kimi-k2.7-code");
    expect(html).toContain("disabled=\"\"");
    expect(html).toContain('aria-pressed="true"');
    expect(html.match(/required=""/g)).toHaveLength(3);
  });

  it("delegates profile management without mounting an internal dialog when onManage is provided", async () => {
    const onManage = vi.fn();
    const container = document.createElement("div");
    const root = createRoot(container);
    await act(() => {
      root.render(
        <I18nProvider initialLanguage="en">
          <LLMProfileSelect
            profiles={[profile("default", "config", false, true)]}
            value="default"
            onManage={onManage}
            onChange={() => undefined}
            onCreate={async () => true}
            onUpdate={async () => true}
            onDelete={async () => true}
          />
        </I18nProvider>
      );
    });

    const manageButton = container.querySelector<HTMLButtonElement>('button[title="Manage"]');
    expect(manageButton).not.toBeNull();
    await act(() => manageButton?.click());

    expect(onManage).toHaveBeenCalledOnce();
    expect(container.querySelector('[role="dialog"]')).toBeNull();
    await act(() => root.unmount());
  });

  it("opens the built-in manager when no external management callback is provided", async () => {
    const container = document.createElement("div");
    const root = createRoot(container);
    await act(() => {
      root.render(
        <I18nProvider initialLanguage="en">
          <LLMProfileSelect
            profiles={[profile("default", "config", false, true)]}
            value="default"
            onChange={() => undefined}
            onCreate={async () => true}
            onUpdate={async () => true}
            onDelete={async () => true}
          />
        </I18nProvider>
      );
    });

    const manageButton = container.querySelector<HTMLButtonElement>('button[title="Manage"]');
    await act(() => manageButton?.click());

    expect(container.querySelector('[role="dialog"]')).not.toBeNull();
    expect(container.textContent).toContain("Model profiles");
    await act(() => root.unmount());
  });

  it("rejects invalid numeric profile values without truncating them", () => {
    expect(parseProfileNumber("4", { integer: true, minimum: 0 })).toBe(4);
    expect(parseProfileNumber("", { integer: true })).toBeNull();
    expect(() => parseProfileNumber("1.5", { integer: true })).toThrow(/integer/);
    expect(() => parseProfileNumber("Infinity")).toThrow(/finite/);
    expect(() => parseProfileNumber("-0.1", { minimum: 0 })).toThrow(/range/);
    expect(() => parseProfileNumber("0", { minimum: 0, exclusiveMinimum: true })).toThrow(/range/);
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
    safety_identifier_env: null,
    prompt_cache_retention: null,
    responses_previous_response_id: null,
    parallel_tool_calls: null,
    auto_wait_on_empty_tool_calls: null,
    fallback_json_actions: null,
    temperature: null,
    max_tokens: null,
    context_window_tokens: null,
    allow_custom_base_url: source === "user",
    source,
    editable,
    is_default: profileId === "default"
  };
}
