import { describe, expect, it } from "vitest";
import { databaseTargetFromRenderer } from "./database.js";

describe("databaseTargetFromRenderer", () => {
  it("allows the default runtime database selector", () => {
    expect(databaseTargetFromRenderer(undefined)).toBe("local");
    expect(databaseTargetFromRenderer("")).toBe("local");
    expect(databaseTargetFromRenderer(" local ")).toBe("local");
  });

  it("rejects renderer-provided database paths", () => {
    for (const candidate of ["/tmp/agent.db", "../agent.db", "foo.sqlite"]) {
      expect(() => databaseTargetFromRenderer(candidate)).toThrow(/Open SQLite database dialog/);
    }
  });
});
