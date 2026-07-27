import { describe, expect, it } from "vitest";
import { RequestEpoch } from "./requestEpoch";

describe("RequestEpoch", () => {
  it("allows only the latest Explain request to publish a response", () => {
    const epoch = new RequestEpoch();
    const first = epoch.begin();
    const second = epoch.begin();

    expect(epoch.isCurrent(first)).toBe(false);
    expect(epoch.isCurrent(second)).toBe(true);
  });

  it("invalidates an outstanding request during refresh or unmount", () => {
    const epoch = new RequestEpoch();
    const request = epoch.begin();

    epoch.invalidate();

    expect(epoch.isCurrent(request)).toBe(false);
  });
});
