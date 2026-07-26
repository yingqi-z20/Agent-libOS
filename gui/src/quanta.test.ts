import { describe, expect, it } from "vitest";
import { parseOptionalQuanta, parseQuantaDraft } from "./quanta";

describe("parseOptionalQuanta", () => {
  it("accepts only positive safe integers", () => {
    expect(parseOptionalQuanta("")).toBeNull();
    expect(parseOptionalQuanta(" 3 ")).toBe(3);
    expect(parseOptionalQuanta("0")).toBeNull();
    expect(parseOptionalQuanta("-1")).toBeNull();
    expect(parseOptionalQuanta("1.5")).toBeNull();
    expect(parseOptionalQuanta("abc")).toBeNull();
  });

  it("preserves invalid drafts while distinguishing them from unlimited", () => {
    expect(parseQuantaDraft("")).toEqual({ raw: "", value: null, valid: true });
    expect(parseQuantaDraft("1.5")).toEqual({ raw: "1.5", value: null, valid: false });
    expect(parseQuantaDraft("abc")).toEqual({ raw: "abc", value: null, valid: false });
  });
});
