import { describe, expect, it } from "vitest";
import { resolveInitialLanguage, translate } from "./i18n";

describe("i18n", () => {
  it("uses a persisted language before navigator language", () => {
    expect(resolveInitialLanguage("en", "zh-CN")).toBe("en");
    expect(resolveInitialLanguage("zh-CN", "en-US")).toBe("zh-CN");
  });

  it("follows Chinese navigator languages when no valid persisted value exists", () => {
    expect(resolveInitialLanguage(null, "zh-Hans-CN")).toBe("zh-CN");
    expect(resolveInitialLanguage("fr", "en-US")).toBe("en");
  });

  it("interpolates translated messages", () => {
    expect(translate("zh-CN", "user.llmCalls", { count: 3 })).toBe("3 次 LLM 调用");
    expect(translate("en", "user.llmCalls", { count: 3 })).toBe("3 LLM calls");
  });

  it("falls back to English or the key for missing translations", () => {
    expect(translate("zh-CN", "top.spawn")).toBe("新进程");
    expect(translate("zh-CN", "missing.translation.key")).toBe("missing.translation.key");
  });
});
