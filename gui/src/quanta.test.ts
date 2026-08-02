import { describe, expect, it } from "vitest";
import { parseQuantaDraft } from "./quanta";

describe("parseQuantaDraft", () => {
  it("accepts only positive safe integers", () => {
    expect(parseQuantaDraft("").value).toBeNull();
    expect(parseQuantaDraft(" 3 ").value).toBe(3);
    expect(parseQuantaDraft("0").value).toBeNull();
    expect(parseQuantaDraft("-1").value).toBeNull();
    expect(parseQuantaDraft("1.5").value).toBeNull();
    expect(parseQuantaDraft("abc").value).toBeNull();
  });

  it("preserves invalid drafts while distinguishing them from unlimited", () => {
    expect(parseQuantaDraft("")).toEqual({ raw: "", value: null, valid: true });
    expect(parseQuantaDraft("1.5")).toEqual({ raw: "1.5", value: null, valid: false });
    expect(parseQuantaDraft("abc")).toEqual({ raw: "abc", value: null, valid: false });
  });
});
