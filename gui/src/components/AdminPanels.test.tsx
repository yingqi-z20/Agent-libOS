import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { I18nProvider } from "../i18n";
import { AppNotices, LoadingScreen } from "./AppNotices";
import { checkpointLabel } from "./CheckpointPanel";
import { capabilityIdentity } from "./CapabilityPanel";
import { parseObjectInput } from "./ObjectTasksPanel";
import { parseJsonInput, remoteOperationIds } from "./RemoteRegistryPanel";

describe("GUI administration helpers", () => {
  it("validates structured arguments locally before remote or task calls", () => {
    expect(parseObjectInput('{"path":"result.txt"}')).toEqual({ path: "result.txt" });
    expect(() => parseObjectInput("[]")).toThrow(/object/);
    expect(parseJsonInput("[1, 2]", false)).toEqual([1, 2]);
    expect(() => parseJsonInput("[]", true)).toThrow(/object/);
  });

  it("builds checkpoint labels without assuming optional metadata", () => {
    expect(checkpointLabel({ checkpoint_id: "cp_1", pid: "pid_1" })).toBe("cp_1");
    expect(checkpointLabel({ checkpoint_id: "cp_2", pid: "pid_1", reason: "before exec" })).toContain("before exec");
  });

  it("uses the runtime's cap_id field as the capability route identity", () => {
    expect(capabilityIdentity({
      cap_id: "cap_1",
      subject: "pid_1",
      resource: "object:report",
      rights: ["read"]
    })).toBe("cap_1");
  });

  it("derives method and tool choices from redacted registry summaries", () => {
    expect(remoteOperationIds("jsonrpc", {
      endpoint_id: "weather",
      methods: [{ method_id: "forecast" }, { method_id: "alerts" }]
    })).toEqual(["forecast", "alerts"]);
    expect(remoteOperationIds("mcp", {
      server_id: "tools",
      tools: [{ tool_id: "search" }, { not_a_tool: true }]
    })).toEqual(["search"]);
  });

  it("renders recoverable loading and stale-data notices", () => {
    const loading = renderToStaticMarkup(
      <I18nProvider initialLanguage="en"><LoadingScreen error="connection refused" onRetry={() => undefined} /></I18nProvider>
    );
    expect(loading).toContain("Could not open the runtime");
    expect(loading).toContain("Retry");

    const notices = renderToStaticMarkup(
      <I18nProvider initialLanguage="en">
        <AppNotices
          error={null}
          snapshot={null}
          streamStatus="reconnecting"
          refreshing={false}
          onDismissError={() => undefined}
          onRetry={() => undefined}
        />
      </I18nProvider>
    );
    expect(notices).toContain("displayed data may be stale");
    expect(notices).toContain('role="status"');
  });
});
