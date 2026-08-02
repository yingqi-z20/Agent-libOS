import { describe, expect, it, vi } from "vitest";
import { persistLanguage, resolveInitialLanguage, syncDocumentLanguage, translate } from "./i18n";

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
    expect(translate("zh-CN", "image.requiredCaps", { count: 3 })).toBe("3 个 required caps");
    expect(translate("en", "image.requiredCaps", { count: 3 })).toBe("3 required caps");
  });

  it("falls back to English or the key for missing translations", () => {
    expect(translate("zh-CN", "top.spawn")).toBe("新进程");
    expect(translate("zh-CN", "missing.translation.key")).toBe("missing.translation.key");
  });

  it("synchronizes the document language exactly", () => {
    const documentValue = { documentElement: { lang: "" } } as unknown as Document;
    syncDocumentLanguage("zh-CN", documentValue);
    expect(documentValue.documentElement.lang).toBe("zh-CN");
    syncDocumentLanguage("en", documentValue);
    expect(documentValue.documentElement.lang).toBe("en");
  });

  it("treats throwing storage methods as unavailable", () => {
    const selectedStorage = { setItem: vi.fn(() => { throw new Error("disabled"); }) } as unknown as Storage;
    expect(persistLanguage("zh-CN", selectedStorage)).toBe(false);
    expect(selectedStorage.setItem).toHaveBeenCalledOnce();
  });
});
