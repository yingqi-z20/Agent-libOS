import { describe, expect, it } from "vitest";
import type { RuntimeProcess } from "./api/types";
import {
  shortProcessId,
  taskDisplayLabel,
  taskLabelFromGoal,
  taskLabelsForStorage,
  taskLabelsFromStorage
} from "./taskPresentation";

describe("task presentation", () => {
  it("turns a multiline goal into a bounded human-readable label", () => {
    expect(taskLabelFromGoal("  Review the project\n\nand fix the highest-impact issue.  ")).toBe(
      "Review the project and fix the highest-impact issue."
    );
    const label = taskLabelFromGoal("改".repeat(100));
    expect(Array.from(label)).toHaveLength(72);
    expect(label.endsWith("…")).toBe(true);
  });

  it("uses a compact process fallback without losing the identifying suffix", () => {
    expect(shortProcessId("pid_936b342f4e4848fb")).toBe("pid_936b342f…48fb");
    expect(shortProcessId("pid_short")).toBe("pid_short");
  });

  it("prefers the session task label over a process id", () => {
    const process = { pid: "pid_936b342f4e4848fb" } as RuntimeProcess;
    expect(taskDisplayLabel(process, { [process.pid]: "Audit the GUI" })).toBe("Audit the GUI");
    expect(taskDisplayLabel(process, {})).toBe("pid_936b342f…48fb");
  });

  it("round-trips bounded session labels and rejects malformed storage", () => {
    const labels = taskLabelsFromStorage(taskLabelsForStorage({
      pid_1: "  Review\nthis project  ",
      pid_2: "改".repeat(100),
      pid_empty: "   "
    }));

    expect(labels.pid_1).toBe("Review this project");
    expect(Array.from(labels.pid_2)).toHaveLength(72);
    expect(labels).not.toHaveProperty("pid_empty");
    expect(taskLabelsFromStorage("not-json")).toEqual({});
    expect(taskLabelsFromStorage("[]")).toEqual({});
  });
});
