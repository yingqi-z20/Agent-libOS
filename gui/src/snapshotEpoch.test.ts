import { describe, expect, it } from "vitest";
import { SnapshotEpoch } from "./snapshotEpoch";

describe("SnapshotEpoch", () => {
  it("rejects an HTTP snapshot when SSE advanced after the request began", () => {
    const epoch = new SnapshotEpoch();
    const request = epoch.beginHttpRequest();

    epoch.acceptStreamSnapshot();

    expect(epoch.acceptHttpResponse(request)).toBe(false);
  });

  it("accepts an HTTP snapshot when no newer source advanced", () => {
    const epoch = new SnapshotEpoch();
    const request = epoch.beginHttpRequest();

    expect(epoch.acceptHttpResponse(request)).toBe(true);
    expect(epoch.acceptHttpResponse(request)).toBe(false);
  });

  it("invalidates pending HTTP responses on an authoritative source replacement", () => {
    const epoch = new SnapshotEpoch();
    const request = epoch.beginHttpRequest();

    epoch.acceptAuthoritativeSnapshot();

    expect(epoch.acceptHttpResponse(request)).toBe(false);
  });
});
